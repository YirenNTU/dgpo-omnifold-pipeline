# Minimal DGPO accumulation toy — protocol v1

## Completed: DGPO after Fourier truth pretraining (2026-09-22)

User explicitly authorized this local launch; offline W&B `g878cw0z`, output
`artifacts/dgpo_toy/cube_pretrained_dgpo_v1`. All three3000step fits completed
(503.8s). Independent reward gains .01037/.00602/.01768, all lower95%>0;
modeTV .05134/.04301/.06106 -> .03596/.03411/.03515; moment -> .392/.395/.389
(truth.4), all corner guards pass. Nine saved endpoint panels independently
recompute; source/reward unchanged and AdamW clocks3000. Some marginal errors
remain/worsen. See output RESULTS.md. This confirms residual refinement after
conditioning-aware pretraining, not an ESS-causal test or production solution.
Use the three selected raw Fourier
checkpoints from `cube_truth_fourier_v1` directly, without resetting their learned
condition adapters. Fixed reward is log binned truth probability minus log
independently estimated initial-generator mode probability; it is not the old
configured-mixture ratio and is not an exact neural density ratio. Frozen source
reference, native DGPO, velocity-MSE coefficient 1, 3,000 steps per seed. Reward
absorption, conditional joint improvement and shape preservation are reported
separately. No classifier or additional pretraining. 55 tests passed, including
interrupted-run recovery; all three real source checkpoint tensors match.
See [protocol and local launch command](CUBE_PRETRAINED_DGPO_PROTOCOL.md).

## Completed: Fourier from the first truth-pretraining update (2026-09-22)

User explicitly authorized local launch; offline W&B `mr15pebb`, output
`artifacts/dgpo_toy/cube_truth_fourier_v1`. All six fits early-stopped (114s,
96000updates). Raw jointTV .261 versus Fourier .070–.082 across all three
seeds; matched-budget and final joint/shape gates pass. Conditional moment
recovers from approximately0 to .349–.367 (truth .4); not full closure.
Read the output RESULTS.md for validation, bootstrap caveat, and limitations.
Raw versus condition-Fourier diffusion from random initialization,
same fixed complete-truth cube data (131072/8192/8192), paired initialization and
batch/time/noise, three training seeds. No classifier, DGPO, KL or EMA. Early
stop20/min_delta1e-4 without a total-step cap; report both equal-budget and
independently early-stopped endpoints. Primary is conditional eight-mode TV,
not mean denoising loss alone. Known k8 basis is an explicit toy inductive bias.
See [protocol and local launch command](CUBE_TRUTH_FOURIER_PROTOCOL.md).

## Same-data H4 on truth-trained diffusion

The new H4 fit uses EXACTLY the saved diffusion truth rows and one fixed DDIM20
negative per same condition. No streaming truth.32768train pairs/8192val pairs/
16384test pairs; existing joint3 architecture, validation BCE early stopping,
raw weights. See [matched H4 protocol](MATCHED_H4_PROTOCOL.md).
Completed: early stop24, selected raw epoch4/update512; held-out test
BCE.425024/AUC.899245/ESS fraction.00053405. Generalizing residual remains,
but ratio concentration/normalization are poor. No DGPO run launched.

## Current: full-truth diffusion pretraining (2026-09-21)

User requested normal sample-target diffusion training: prepare a FIXED complete
truth dataset, train from random initialization, stop by validation MSE. No
Gaussian teacher, classifier or DGPO. The proposed20k step limit was superseded
before execution. Train32768/val8192/test16384; patience20/min_delta1e-4; raw weights.
See [truth training protocol and command](TRUTH_PRETRAIN_PROTOCOL.md).
Completed: early stop epoch30/update1920; best raw epoch25/update1600.
Val MSE1.08136->1.00192; selected test1.00066. Joint moment improvement remains
unresolved. Dataset and outputs under truth_dataset_seed17_v1 and
truth_diffusion_earlystop_v1 respectively; neither overwrites an older run.
Previous Gaussian-baseline DGPO experiments below retain their distinct scope.

## Saved no-Fourier classifier ablation (2026-09-21)

The user-selected plain classifier is loaded directly from
`conditional_fourier_fit10k_v1/seed17/models.pt` (validation-selected9900).
No new classifier fitting, no Fourier input channels, unchanged stored CDF
normalization. The initial direct endpoint-KL1 run was interrupted on user
direction at1248, with its full valid state preserved. The current requested
arm restarts at the original pretrained policy with velocity-MSE coefficient1,
matched against the existing strong-H4 velocity10k control.
The strong fixed H4 judge and known high-order moments evaluate both policies.
See [current proxy protocol and command](PLAIN_VELOCITY_PROTOCOL.md) and
[original classifier/setup protocol](WEAK_CLASSIFIER_PROTOCOL.md).
The earlier100-fit-step weak-classifier draft was never run. This saved-model
comparison also changes training-data regime/capacity, not just Fourier or ESS.

## Direct conditional endpoint-KL experiment (2026-09-21)

The latest arm subtracts a directly evaluated DDIM endpoint log-density ratio
from the fixed H4 reward BEFORE the original DGPO advantage/gate. No learned
KL critic, velocity penalty, or pathwise KL gradient. See
[predeclared protocol](NET_REWARD_KL_PROTOCOL.md). It requires a sufficient
global inverse certificate plus checked float64 inverse/Jacobian calculations;
it stops on failed validity and preserves the last validated full state.

Completed10k result: [DIRECT_ENDPOINT_KL_RESULTS.md](DIRECT_ENDPOINT_KL_RESULTS.md).
Valid numerical run; no decisive reward/high-order improvement at this budget.
User-authorized continuation to20k: [extension protocol/command](DIRECT_ENDPOINT_KL_EXTENSION_PROTOCOL.md).
It preserves full training state and compares20k against the saved10k policy
on a fresh endpoint panel; historical controls remain labelled10k.

```bash
python experiments/dgpo_toy/net_reward_kl.py \
  --source artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt \
  --comparator artifacts/dgpo_toy/fixed_reward_joint3_resume10k_v1 \
  --velocity-comparator artifacts/dgpo_toy/velocity_kl1_joint3_10k_v1 \
  --output artifacts/dgpo_toy/direct_endpoint_kl1_joint3_10k_v1 \
  --policy-steps 10000 --coefficient 1
```

`--coefficient 0` turns the subtraction off (density checks still run);
the original fixed-reward runner is the cheaper no-KL control. Optional
`--resume-from PATH` restores model, AdamW, random stream and density clock.
Use a different output for a resumed run. Progress is streamed to JSONL;
report and full state are saved every1000 steps and at completion/validity stop.
Do not rerun the command against a running output directory.

The remaining sections record earlier experiments, not the current arm.

This local CPU experiment asks whether the production DGPO surrogate can have
a useful initial direction but accumulate at a distribution different from
truth even with an **exact frozen reward and exact endpoint KL**. It does not
claim to reproduce every aspect of H4, or diagnose the real run by analogy.

Completed local measurements and interpretation: [RESULTS.md](RESULTS.md).

The user's follow-up asks for **low ESS and slow pure-reward learning without
additive KL**. That distinct experiment is implemented in [narrow.py](narrow.py):
[protocol](NARROW_PROTOCOL.md), [completed results](NARROW_RESULTS.md).
Do not substitute the original Gaussian/KL result for that revised question.

## Conditional neural diffusion + learned Fourier reward

The subsequent question requires genuine conditioning, many trainable parameters,
and higher-order dependence rather than a scalar variance. Implemented separately
in [conditional.py](conditional.py); contract: [CONDITIONAL_PROTOCOL.md](CONDITIONAL_PROTOCOL.md).
Executed pilot results and their limitations: [CONDITIONAL_RESULTS.md](CONDITIONAL_RESULTS.md).

The finite-pool pilots did not pass the low-ESS gate. The separate
[low_ess_recovery.py](low_ess_recovery.py) experiment changes only the classifier
fitting stream to fresh actual-generator/truth examples, and adds independent
tail and numerical neural-DDIM density diagnostics. It preserves the previous
source generator and all original gates. See
[LOW_ESS_RECOVERY_PROTOCOL.md](LOW_ESS_RECOVERY_PROTOCOL.md).
Completed outcomes: [LOW_ESS_RECOVERY_RESULTS.md](LOW_ESS_RECOVERY_RESULTS.md).

The user subsequently authorized testing the **fixed approximate learned reward**
despite the earlier ratio-fidelity failures. No classifier retraining is needed:

```bash
python experiments/dgpo_toy/fixed_reward.py \
  --source artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt \
  --output artifacts/dgpo_toy/fixed_reward_joint3_v1 --seed 17
```

[Fixed-reward protocol](FIXED_REWARD_PROTOCOL.md) and
[completed300-step results](FIXED_REWARD_RESULTS.md): DGPO and pathwise both
plateau in the pilot. This is not proof of low-ESS causality or truth closure.

For the separately declared tenfold length extension, replay the same short
trajectory and verify exact weights/history at its endpoint before continuing:

```bash
python experiments/dgpo_toy/fixed_reward.py \
  --source artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt \
  --output artifacts/dgpo_toy/fixed_reward_joint3_long_v1 --seed 17 \
  --policy-steps 3000 --replay-from artifacts/dgpo_toy/fixed_reward_joint3_v1 \
  --endpoint-seed 92017
```

[Long-run protocol](LONG_REWARD_PROTOCOL.md) and
[completed3000-step results](LONG_REWARD_RESULTS.md). The old300-step files contain
weights only; this command replays from the initial source without resetting
AdamW at step300. New `*_state.pt` files retain model, AdamW, training RNG and
history at1000-step intervals, the replay boundary and the endpoint. They are
full-state snapshots. Direct continuation is now supported with `--resume-from`:

```bash
python experiments/dgpo_toy/fixed_reward.py \
  --source artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt \
  --output artifacts/dgpo_toy/fixed_reward_joint3_resume10k_v1 --seed 17 \
  --policy-steps 10000 --resume-from artifacts/dgpo_toy/fixed_reward_joint3_long_v1 \
  --endpoint-seed 93017
```

This applies only steps3001–10000, restoring AdamW and RNG rather than replaying
the prefix. [Continuation protocol](RESUME_10K_PROTOCOL.md) and
[completed10000-step results](RESUME_10K_RESULTS.md). Both methods improve the
fixed reward; DGPO starts late and subsequently shows recoil. The original gate
reference remains fixed at step0. Resumed and uninterrupted training are tested
for exact agreement in model, optimizer, RNG and history for both estimators.

Render the saved histories without reevaluating or training any model:
`python experiments/dgpo_toy/plot_fixed_reward.py artifacts/dgpo_toy/fixed_reward_joint3_resume10k_v1`

```bash
python experiments/dgpo_toy/low_ess_recovery.py \
  --source artifacts/dgpo_toy/conditional_fourier_v1/seed17/models.pt \
  --output artifacts/dgpo_toy/low_ess_recovery_v1 --seed 17
```

Default classifier budget is20,000 updates; add `--smoke` for an execution check.
`reward.pt` saves the best-validation-BCE frozen reward plus source generator;
`progress.jsonl` streams progress and `report.json` records phase results,
including partial results on errors. The numerical inverse/Jacobian diagnostic
also checks a sufficient global-bijection bound for this exact neural architecture;
if the bound fails, density interpretation remains single-branch only. It never
supplies classifier training labels or policy rewards.

The fresh-coordinate run did not reach low ESS. A separately declared generic
all-triples Fourier representation follow-up is available (not parameter matched):

```bash
python experiments/dgpo_toy/low_ess_recovery.py \
  --source artifacts/dgpo_toy/conditional_fourier_v1/seed17/models.pt \
  --output artifacts/dgpo_toy/low_ess_joint3_v1 --seed 17 --feature-mode joint3
```

```bash
python experiments/dgpo_toy/conditional.py --smoke --output artifacts/dgpo_toy/conditional_smoke
python experiments/dgpo_toy/conditional.py --output artifacts/dgpo_toy/conditional_fourier_v1 --seeds 17
python experiments/dgpo_toy/conditional.py --output artifacts/dgpo_toy/conditional_fourier_replication --seeds 17 29 43
python -m pytest -q experiments/dgpo_toy
```

Full mode actually pretrains an MLP denoiser on an analytic baseline v teacher,
fits plain and Fourier classifiers on generated-vs-truth samples, then freezes
the best-validation-BCE Fourier critic. It runs DGPO and a separately labeled
pathwise diagnostic ONLY if the pretrained marginal, classifier and low-ESS
validity gates pass. Smoke exercises both paths without scientific inference.
The control is not production DGPO or compute matched. The classifier is given
individual marginal-CDF coordinates and harmonics, never the true joint formula.

Per seed: live `progress.jsonl`, `report.json`, initial generator and both selected
classifiers in `models.pt`, and final arm weights if training was eligible.
`summary.json` combines seeds. CPU-only, one PyTorch thread, no W&B or NERSC setup.
Commands overwrite their named output files, not unrelated files. A previous
failed run's `failure.json` remains as history; use `report.json`/`summary.json`
and current `progress.jsonl` for the latest completion state.

## Frozen contract (declared before the first run)

- Two-dimensional zero-mean Gaussian. Reference and truth have exactly the
  same one-dimensional marginals; truth correlation is 0.8. This is a joint
  correlation problem, not yet an irreducible higher-order dependence problem.
- Only two trainable log-variance parameters, in the fixed 45-degree eigenbasis.
  The denoiser is the analytic Gaussian v-predictor. Model capacity and learned
  classifier error are removed. Every distribution has full support.
- Raw deterministic DDIM20 with the repo's finite-endpoint cosine VP schedule.
  A linear DDIM chain collapses exactly to two scale factors; the runner uses
  those factors, and tests compare them against the stepwise chain. Measurements
  and reward densities use the **actual finite-step endpoint distribution**,
  not the nominal Gaussian denoiser covariance.
- Frozen reward `log p_truth - log q_reference`; no tempering, clipping, refit,
  EMA, learned critic, features, neural network or external service.
- Import the unchanged `compute_per_event_advantage` and `build_dgpo_loss`
  functions directly from production source via AST, avoiding its unrelated
  Lightning/Ray import tree. No alternative loss implementation is silently used.
- K=8, LOO unscaled, beta_dgpo=1, eight independent timestep/noise draws per
  optimizer step, uniform t in [0,0.7], shared noise/time across a candidate
  group, and mean per-coordinate velocity MSE. Gate is formed per timestep
  before averaging. Rollouts and advantages are detached in the DGPO arm.
- AdamW, weight decay 0.001, clipping norm 1. Toy LR=0.01 is deliberately in
  this two-parameter coordinate system, **not** a claim to copy production LR.

## Matched velocity-MSE coefficient 1, 10000 updates

One new step-0 DGPO arm; reuse the existing no-KL control. This is the legacy
`.5 * mean((v_current-v_step0)^2)` surrogate, not exact endpoint KL. No classifier
refit, optimizer change or production edits. Compare fixed reward AND higher-order
structure against truth, with a new paired endpoint panel:

```bash
python experiments/dgpo_toy/velocity_kl.py \
  --source artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt \
  --comparator artifacts/dgpo_toy/fixed_reward_joint3_resume10k_v1 \
  --output artifacts/dgpo_toy/velocity_kl1_joint3_10k_v1 \
  --coefficient 1 --policy-steps 10000 --seed 17 --endpoint-seed 94017
python experiments/dgpo_toy/plot_velocity_kl.py artifacts/dgpo_toy/velocity_kl1_joint3_10k_v1
```

See [protocol](VELOCITY_KL_PROTOCOL.md) and [completed results](VELOCITY_KL_RESULTS.md).
No-KL passes the intended reward-transfer control; coefficient1 suppresses
learning and has not achieved truth alignment. Unconstrained no-KL marginal
drift is expected and does not veto its transfer result.
Full-network component gradients are
pre-Adam/pre-clipping; original head-gradient ESS still describes the main term.

## Three arms, one question

1. `exact_reward_kl`: exact `-E_q[reward] + KL(q||q_ref)`; its gradient must
   equal `KL(q||p_truth)`. Positive control for representability/optimizer.
2. `dgpo_endpoint_kl`: production DGPO main gradient + exact endpoint KL,
   coefficient 1. This isolates the main surrogate from learned KL-critic error.
3. `dgpo_only`: same production main gradient without additive KL. This
   distinguishes absent regularization/overshoot from the composite equilibrium;
   maximizing a fixed reward alone is not expected to recover truth.

All arms start from the same reference and use matching rollout/noise streams.
The exact arm integrates expectations analytically; it is an oracle, not a
compute-matched stochastic estimator. Independent evaluation normals are never
used for updates. Two-dimensional covariance alone is sufficient for exact KL.

## Measurements and predeclared decisions

Primary: exact final `KL(q_endpoint || truth)` and its last-20%-of-updates mean.
Also log initial/best KL, reward mean/std, correlation/marginal variance,
main/KL gradient vectors and cosine, gradient-to-true-KL alignment, cancellation,
gate saturation, optimizer displacement and resulting *exact* KL change.
Fixed-oracle and current Bayes-optimal AUC/BCE are secondary Monte Carlo checks;
there is no cold-classifier training budget ambiguity.

Default: 600 updates, seeds17/29/43, batch256 event groups. First smoke can use
one seed; its conclusions remain exploratory. No early stopping or best-test
checkpoint selection. Stop an arm on nonfinite values and mark it invalid.

Operational reproduction gate, evaluated separately per seed:

- Positive control final KL <=1% of initial KL.
- DGPO+KL first10 updates reduce exact KL by >=0.1% of initial KL.
- DGPO+KL late-window mean KL remains >=10% of initial KL.
- Its second-half-of-late-window improvement is <=1% of initial KL (plateau).

All three seeds must pass to label the default experiment a replicated toy
failure. Otherwise say `not_reproduced` or `inconclusive`, not force failure by
increasing target difficulty or reducing training budget. A failure here is a
**sufficient toy mechanism**, not proof of the production cause. In particular,
this cannot reproduce a failure that requires conditional context, high-order
structure, missing effective coverage, learned critics, or neural capacity.

## Local usage

From the repository root (Python with torch, numpy and matplotlib):

```bash
python experiments/dgpo_toy/run.py --output artifacts/dgpo_toy/baseline
python -m pytest -q experiments/dgpo_toy/test_toy.py
```

The runner overwrites only its named report/CSV/PNG files in the chosen output,
not production files. Use a new output directory for each scientific variant.
No W&B, Ray, NERSC, dataset or checkpoint is needed. Report records full settings.

## Round 2: gradient-unit check (declared after v1 smoke, before this probe)

Smoke seed17 reproduced the predeclared pattern: exact-control final KL
6.21e-7, DGPO+exact-KL 0.7183, initial KL1.267. The gate remained about0.48;
the main and KL gradients nearly opposed at the stalled point. This does not
alone prove the penalty is wrong: exact reward and exact KL must also cancel
**at the correct target**.

Next single question: is one fixed scalar matching the initial DGPO gradient
to the exact reward gradient sufficient? Use four independent 8192-group
gradient panels at the initial policy (not endpoint selection) to calibrate
the positive least-squares scale, then freeze it for all policy updates. Only
the external DGPO main multiplier changes; do not alter its internal gate.
This is an explicitly changed toy objective, not target-preserving production
tuning. Keep the original run and endpoint rule.

Also measure the expected vector field at the exactly representable truth.
The exact reward+KL gradient must be zero there. Estimate production main
gradient with four independent panels and report mean/SE; report residuals
both at scale1 and at the best positive scalar **at that point**. The latter
is a diagnostic lower bound, not a deployed/test-selected scale. A significant
component orthogonal to the KL gradient would show scalar scaling cannot make
truth stationary in this two-parameter toy. Do not infer it from noisy cosine
alone. This removes AdamW and training length from that particular question.
