# Minimal reward-gain-without-closure reproduction

Status: launched locally on 2026-09-26 after the user's explicit "start training"
authorization for this batch. Local toy only; no production changes or NERSC
upload. W&B run: `99jgjjnw` (launch record below).

## Question and first-principles boundary

Can changing only the learned reward's positive within-mode shape reintroduce
the real-case signature: fixed H4 reward rises, but adequately trained fresh H4
discrimination does not improve, despite functioning nonlinear conditioning?

`227e4975` supplies the observed signature, not causal proof: within-round reward
improves, cold AUC .90618 -> .90995 by step300, audit validation BCE .40748 ->
.38149, fits1452-2040. Its angle+momentum network and audit protocol differ from
the previous angle-only production run. We do not require toy AUC to equal .91.

The successful `learned_film_conditioning_v1` and `film_auc_trajectory_v1` used
shape-matched H4, not the original full-truth H4. That distinction is the one
limiting factor tested here. Existing first-add reward-target comparisons did
not test this boundary under the now-working nonlinear FiLM architecture.

## Small intuitive conditional distribution

- Observed continuous c in [-1,1]; output y has three coordinates.
- Eight noisy cube corners. Truth widths are .15; c controls positive versus
  negative triple-parity probability through four localized Gaussian bumps
  passed through tanh, NOT a sinusoidal target.
- Every ideal single/pair marginal stays unchanged; the three-way dependency
  changes. The actual trained generator also has small within-corner shape
  errors; it is NOT analytically perfect at all lower orders.
- Source is the existing ordinarily velocity-pretrained uniform-cube diffusion,
  `artifacts/dgpo_toy/nonperiodic_cube_classifier_extended_v1/reference.pt`.
  No generator retraining or DGPO endpoint initialization.

Both arms use exactly the successful c-only nonlinear FiLM: Fourier c frequencies
[1,2,4,8] -> 8/32/32 SiLU encoder -> encoder-only LayerNorm -> three zero-init
scale/shift heads; pretrained backbone128. No truth modes/formula in model input.

## Exactly two paired reward targets

| Arm | Positive target | Negative target |
| --- | --- | --- |
| full | Truth mode probabilities AND truth within-mode shape | Initial generator |
| shape | Same exact sampled truth mode IDs, initial generator within-mode shape | Same exact initial-generator samples |

Refit BOTH H4 rewards here so sampling/fit procedures match. Do not compare the
old continuous-panel full critic with the old grid-panel shape critic and call
that one changed variable. Both use raw c/y plus coordinate Fourier k1..4, same
128x3 GELU architecture, initialization23, minibatch RNG40023, balanced batch512,
AdamW3e-4, WD.001, clip1. Joint fit budget: minimum2000, check100, patience20,
min_delta1e-4, maximum16000. Each selects exact minimum validation BCE. Test
does not select weights. Require both plateau and test AUC>.55/BCE<ln2-.01;
this observed-test validity screen is exploratory, not confirmatory selection.

Use existing audited conditional swaps: train128 conditions x256, val128x64,
test256 interleaved conditions x128; independent donor pools with1024 samples
per condition and >=16 per mode. Full and shape positives share exact mode IDs;
negatives, conditions, sample counts, minibatches and initial weights match.
Full truth shapes are Gaussian restricted to their orthant (negligible crossing
mass at width.15). Shape matching is empirical and uses known toy modes; finite
donor resampling/duplicates are limitations. It is not physics-prior-free and
is NOT proposed as the production solution.

## Unchanged policy workflow

Same initial FiLM weights for both arms. Native detached DGPO gate, unscaled
LOO, coefficient1 HALF velocity MSE to the original fixed reference. No exact
endpoint-KL claim. AdamW1e-4, WD.001, clip1, batch64,K8,M4,t~U[0,.7],DDIM50.
Policy seed17, encoder451017, monitor284017. All policy weights train; raw/noEMA.
No refit, reference recentering, optimizer reset, reward transform, LR schedule,
new feature, coefficient sweep, or architecture sweep.

Front-loaded audit steps0/25/50/100/150/200/300/1000. The user prioritizes
the early trajectory and caps this pilot at1000 updates per arm: six positive
milestones through300, then the final1000 checkpoint. Optimizer/RNG/history
and the original reference survive every milestone. Primary endpoint is1000;
do not run or require the previously planned3000 endpoint. A negative1000-step
pilot is not evidence that a longer trajectory can never improve.
The initial policy is identical across arms: fit its cold audit ONCE, then fit
both arms at each of seven positive milestones (15 cold fits total, originally7).
Step0 is shown on both W&B arm trajectories as the same shared baseline, not
as two independently fitted measurements. Classifier fit budgets/stopping rules
are unchanged; denser audits cost more evaluation, not more policy updates.
Earlier points describe the trajectory only. No automatic extensions or seed
search if failure is not reproduced. Do not select the best intermediate point
or treat repeated monitoring as independent evidence for the final decision.

## Measurements that determine the decision

1. Independent fixed-grid all-candidate mean reward; paired context gain/CI.
   Log best-of16 separately. Never substitute winner-only gain for mean gain.
2. Cold full-truth H4 audit at actual step0 and every milestone: independent
   train32768/val8192/test16384 continuous contexts, seed41, same paired truth
   and rollout noise. min2000/max16000, check100,patience20,min_delta1e-4.
   Fit separately to plateau, select validation BCE; final test AUC/BCE untouched.
   Invalid audit is inconclusive, not closure. Initial audit is run once and
   retained; all later audit panels verify identical condition/truth identities.
3. Conditional eight-mode TV, parity error, corner retention and low-order sign
   moments. Secondary diagnostics, never a veto of fresh-classifier closure.
4. Reward gain decomposition: change in mode probabilities evaluated with
   frozen source within-mode mean scores, plus remaining shape/interactions.
   Scores from BOTH rewards are evaluated on BOTH arms. No mode-empty fallback.
5. Existing native gradient traces, gate saturation, within-K/global ESS,
   reward/reference gradient cosine/projection, FiLM scales and branch gradients.

Primary final audit comparisons: full-baseline, shape-baseline, full-shape.
2000 paired-context bootstrap draws, simultaneous95% bands across these three
contrasts separately per metric. Conditional on trained models; not seed
replication or joint AUC/BCE family-wise inference. Reward normal CIs on the
fixed grid are diagnostic, not training-seed uncertainty. No reward gain scale
comparison between separately trained critics.

Predeclared minimal-failure criterion for FULL, with valid audits:

- all-candidate own reward gain lower95 > .01;
- fresh AUC-gap delta lower95 > -.005 AND fresh BCE delta upper95 < +.005,
  excluding an improvement of these prespecified sizes. A nonsignificant delta
  alone is NOT evidence of plateau. Resolved worsening is reported separately.

Matched-control rescue requires shape AUC-gap delta upper95 < -.005, BCE delta
lower95 >0, and full-minus-shape gap lower95 >0.

- Failure + control rescue: supports reward-target package as a contributor.
  Inspect shape/mode decomposition before proposing a mechanism. Changing target
  also changes learned score scale, ESS and approximation errors, so this does
  NOT uniquely prove shape exploitation, KL mismatch or a production cause.
- Full also improves: this proposed factor is not sufficient at this budget;
  do NOT increase difficulty until it fails.
- Both fail / reward does not improve / uncertain fits or intervals: unresolved;
  do not call this isolation of the real mechanism.

## Local commands

Read-only no-fit/no-update preflight:

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_DISABLE_GIT=true \
/opt/miniconda3/envs/MyEve/bin/python -m experiments.dgpo_toy.reward_shape_failure \
  --output artifacts/dgpo_toy/reward_shape_failure_early1k_v1 --preflight
```

Training command used for the authorized local launch:

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_DISABLE_GIT=true \
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.reward_shape_failure \
  --output artifacts/dgpo_toy/reward_shape_failure_early1k_v1 \
  --milestones 25 50 100 150 200 300 1000 \
  --max-fit-steps 16000 --wandb-mode online
```

W&B display name: `Why reward rises without closure? | full vs shape-matched H4 | nonlinear FiLM | V MSE 1`.
Group `Conditional reward transport`; run ID remains separate. Reward-fit and
each cold-audit fit have distinct clocks; policy/validation use policy updates.
Offline is the program default. Outputs preserve all selected classifiers,
milestone policies, optimizer/RNG states, paired panels and final test scores.
Expected local compute is tens of minutes; audit stopping determines duration.
The front-loaded schedule increases cold-fit count from7 to15, so expect materially
more audit time; policy training is now1000 updates per arm. The new output
directory preserves any previously executed sparse experiment unchanged.

## Round record

Outcome: implementation/preflight deliverable, not an experimental finding.
Decision: unresolved until executed. Learning index: no causal decision changed
yet; implementation time must not be counted as successful experimental evidence.
Deleted: new toy/pretraining, plain classifier, oracle reward, no-KL arm, feature
and LR sweeps, repeated production-scale jobs. Next limiting factor: whether
the full-truth H4 changes the successful FiLM trajectory at fixed policy setup.
Better next round: train the paired reward targets on matched panels before
claiming a causal target comparison.

## Implementation verification (2026-09-26)

60 tests passed across this runner and the existing reward transport,
conditioning, FiLM and dense-AUC suites. New tests check exact positive-mode and
negative/context pairing, identical-target fit equivalence, invalid short fits,
non-inferiority versus mere nonsignificance, paired audit intervals, additive
reward decomposition, and exact staged-versus-continuous native policy weights,
AdamW state and RNG. These tiny synthetic tests are not experiment results.

Actual-source read-only preflight passed: initial FiLM velocity outputs and
DDIM samples are bitwise equal to the saved pretrained source;61379 total
parameters,26688 added. Name validation and syntax checks passed. The configured
experiment output directory does not exist; no reward/policy fit was launched.

Earlier dense-audit update (before front-loading): **62 tests passed**. Regression checks compare sparse
and dense staged policy execution with continuous execution, including global
RNG use by intervening audit fits; policy weights, AdamW state and policy RNG
remain identical. Both validation curves include the shared step0 measurement.
Actual-source preflight confirms audit steps0/100/250/500/750/1000/2000/3000 and
15 planned cold fits. Initial velocities/samples still match exactly. The dense
output directory is absent and no experimental training has been started.

Front-loaded revision:62 tests passed again. Actual-source preflight confirms
0/25/50/100/150/200/300/1000/3000 and17 cold fits, with unchanged initial
velocities/samples and unchanged classifier stopping rules. No training launched.

User-requested1000-update cap:62 tests passed again. Default config and the
primary decision endpoint now both end at1000. Preflight confirms audit steps
0/25/50/100/150/200/300/1000 and15 cold fits. This caps DGPO policy updates, not
the independent classifier fit budgets. The early1k output directory is absent;
execution authorization has been requested separately, with no job launched.

## Authorized launch (2026-09-26)

The user subsequently authorized this exact batch with "start training". Launched
the command above locally with online W&B, run `99jgjjnw`:
https://wandb.ai/b11202011-national-taiwan-university/dgpo-toy/runs/99jgjjnw

Output: `artifacts/dgpo_toy/reward_shape_failure_early1k_v1`.
Each arm is capped at1000 policy updates; cold audits remain at
0/25/50/100/150/200/300/1000, with one shared baseline and15 planned fits.
Startup verified online synchronization and paired reward panels. At the last
launch check (176.3 elapsed seconds), the report was `training_dgpo`, active
phase `endpoint`, arm `shape`, policy step25, with no reported error. This
confirms execution, not an experimental conclusion. No remote job was submitted.
