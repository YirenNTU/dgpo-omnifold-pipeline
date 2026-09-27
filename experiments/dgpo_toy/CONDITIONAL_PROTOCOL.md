# Conditional Fourier / low-ESS neural diffusion — protocol v1

Declared before the first run. This is a new experiment, not a reinterpretation
of the scalar `narrow.py` result. Primary question: can production DGPO improve
a frozen learned reward when conditional, genuinely higher-order discrepancies
produce low ESS? No production files or NERSC jobs are changed.

## Distribution and model

- Context c has three continuous coordinates. Output y has 12 coordinates.
  The baseline is a conditional Gaussian with a nonlinear context-dependent
  mean mu(c) and unit marginal variance. All networks share parameters across c.
- Transform each residual individually to a uniform angle with its Gaussian
  marginal CDF. Truth is a 90% structured / 10% baseline mixture. In each of
  four disjoint triples, the sum of three angles has a context-dependent
  von Mises phase (kappa=8). Every conditional univariate AND bivariate marginal
  is exactly the baseline's. The discrepancy requires at least three variables.
  Sampler and nominal log density are known; test their moment identities.
- This coordinate construction is a synthetic modeling choice, not a discovery
  about physics. The classifier receives individual coordinates and context,
  NEVER the triple identities, summed phase, or oracle density.
- A three-hidden-layer MLP v-denoiser is actually trained, then sampled through
  the full 20-step DDIM chain. It is NOT an analytic one-parameter generator.
  Baseline pretraining distills the analytic conditional-Gaussian v teacher,
  avoiding noisy denoising targets as a separate bottleneck. No analytic teacher
  is called in the learned model's forward pass or DGPO update. Teacher nominal
  variance accounts for finite DDIM variance; the achieved neural baseline must
  independently pass marginal/covariance checks. This is teacher pretraining,
  not a simulation of EveNet's complete pretraining history.

## Classifiers and splits

- Fit plain and Fourier classifiers to the SAME actual pretrained-generator
  samples versus truth, paired at each context. Same fit budget/hidden widths;
  Fourier has a larger input projection, so this is not parameter-count matched.
- Both receive individual Gaussian-CDF angle coordinates and context. Fourier
  adds sin/cos harmonics 1..4 of each individual angle, not hand-built joint
  features. Both have a nonlinear encoder and linear logit head.
- After 50 warmup updates, fix encoder-output standardization from TRAIN data
  only, with a function-preserving head reparameterization. Continue learning
  the encoder; never refit the statistics on validation/test or policy rollouts.
- Disjoint train, validation, classifier test, policy monitor, and final endpoint
  context/noise streams. Select classifier by validation BCE, never test AUC/ESS.
  The selected Fourier classifier is frozen for ALL policy updates. Plain is
  a representation control only; do not launch another DGPO arm for it.
- Report BCE, AUC, ratio normalization, ESS, tails, and conditional reward
  variation. Low ESS must be observed, not created by scaling/clipping logits.

## Matched policy arms

1. DGPO: production LOO-unscaled + detached nonlinear gate, beta_dgpo=1,
   detached DDIM rollouts, K8, M4, t uniform in [0,.7], shared noise/time within
   each candidate group, mean-coordinate v MSE, frozen baseline inside gate.
   NO additive KL or reference-trust penalty, no refit, tempering or EMA.
2. Pathwise diagnostic: same network/checkpoint, frozen reward, contexts,
   initial noises, K, AdamW LR/decay and update count, but backpropagate through
   DDIM into the reward. This is an explicitly DIFFERENT gradient estimator,
   not production DGPO and not compute matched. It is an actionability control,
   not a proposed production replacement or proof of truth closure.

Default 300 policy updates, AdamW LR=1e-4, decay=.001, clip=1. Fixed endpoint,
no best-policy selection. Step 100 is secondary. Stop on nonfinite values.
Classifier/pretraining budgets and all settings are saved in report.json.

## Predeclared validity and decision

Full runs skip policy training and return `inconclusive_setup` unless ALL pass:

- Baseline: per-coordinate residual Gaussian KS <=.06; maximum variance error
  <=.15; maximum off-diagonal covariance <=.08; context-bin mean error <=.10.
- Learned Fourier: held-out BCE <=.60, AUC >=.70, ESS/N <=.01;
  abs(log mean exp(logit)) <=.50. These are operational gates, not proof of
  complete tail calibration. Report nominal-oracle ESS separately: oracle
  p/q_Gaussian is NOT p/q_actual_neural and never enters the training reward.
- On independent K8 contexts, at least 1% of groups have reward range >1e-3.
- Truth's known third-order signal exceeds the baseline's by >=.20. Lower-order
  preservation is structural; finite-sample baseline checks guard pretrain error.

Primary: final-minus-initial frozen Fourier reward on a NEW endpoint panel
with paired contexts/noise; 95% normal interval uses context-cluster means,
not candidates as independent events. Predeclared meaningful improvement:
lower CI >0 AND gain >=.10 logit units. If DGPO passes: `reward_improves`.
If DGPO fails but pathwise passes: `dgpo_transfer_deficit` at this tested budget.
If neither passes: `inconclusive_actionability`, not low-ESS causality.
One seed is a pilot; use --seeds 17 29 43 for replication. Do not change gates
or kappa after seeing the result to manufacture a desired failure.

Log reward/std, fixed-weight global ESS and within-K ESS separately, informative
group fraction, gate saturation, gradient norm, clipping, actual AdamW update
norm, and frozen-reward/known-joint monitor trajectories. Monitor points do not
select policy checkpoints. ESS after updates describes FROZEN classifier weights,
not a refitted p/q_current ratio. Pure reward gain is not density closure.

`--smoke` uses tiny budgets and runs both policy arms despite failed setup gates;
it is explicitly `smoke_only` and cannot support a scientific conclusion.
Writes only named files within --output; existing named outputs are overwritten.
No W&B/network dependency: progress is flushed live to the terminal and JSONL.

## Declared follow-up after v1 seed17, before the extended run

The 5000-fit pilot passes all neural-baseline checks. Fourier test BCE=.551569,
AUC=.822912, ESS/N=.072888, log mean ratio=.539323. It fails low-ESS and
normalization gates, so policy updates are NOT eligible (`inconclusive_setup`).
Validation BCE is still improving at the final fit update (.561860 at4900,
.552026 at5000). Preserve this result and the exact original gates.

One bounded follow-up changes ONLY classifier fit budget to10000 for BOTH
plain and Fourier arms, using the same seed/data/init/optimizer/selection.
Default remains5000; use `--classifier-steps 10000` and a separate output.
This is an exploratory budget test, not retroactive success for v1. No logit
rescaling, kappa change, gate relaxation or continued extension is authorized
by this protocol. The learned ratio still must pass original validity gates
before DGPO/pathwise policy training. Keep pretraining and policy budgets fixed.
