# Conditional cube: reward-to-mode-probability diagnosis

## Difficulty increase: reference log-odds x4

cube_difficulty_sharp4_v1 holds targetpositive=.5+.4c, cUniform[-1,1], eight
centers,width.15 and all one-/two-variable conditional marginals unchanged.
Only referencepositive changes to sigmoid(4*logit(.5-.4c)). Same131072 data
budget, random initialization, raw128MLP,DDIM50,training/validation seeds,earlystop.
Actual checkpoint/earlystop steps and categorical reward must change with the
reference; not an ESS-only causal intervention or compute-matched pretrain.
Dataset metadata is in source report; this run does not inherit the easy weights.

After earlystop, measure old and new generators on same new seed720017. Require
actual preferredmass <=half easy and global oracleweightESS <=.75easy, with
nearcorner>=.9, before calling this a successful harder-start intervention.
No reference-fit accuracy gate. If not established, report that, do not call RL
a failure. On pass, same300-update noPenalty/V1 oracle policy runner, with
condition-binned mode probabilities, rewardgain,shape and weight diagnostics.
This stage asks whether successful easycase degrades when reference good modes
become rarer. No classifier, target change, capacity change or hidden transport.

```sh
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.cube_difficulty \
  --output artifacts/dgpo_toy/cube_difficulty_sharp4_v1
```

## User-authorized imperfect-baseline test

The user explicitly corrected the prior stopping rule: imperfect mode probabilities
are intended, not grounds for blocking the reward-transfer diagnostic. Retain the
old gate failures as reference-fit measurements, not as numerical/coverage failure.
cube_oracle_policy.py loads selected continuous4x epoch42 step10752 directly,
does not pretrain, alter weights or install analytic velocity. Source has all
modes and95.9%nearcorner, so it is usable for this explicitly narrower question.
Run cube_oracle_policy_v1:300 updates each, no penalty versus velocityMSE1,
same fresh AdamW/source/reference/training RNG. Reward is the fixed known DATA
mixture categorical log-ratio, not exact ratio to the actual imperfect diffusion.
Primary paired independent reward and preferred-mass gains, also in8condition
bins, with corner shapes and targetTV reported. No target-closure guarantee.
Fresh endpoint690017, physicalmonitor700017,nativemonitor710017. No classifier.
This supersedes gate-based refusal to run this diagnostic; no prior result is
relabeled as having passed its original gate.

```sh
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.cube_oracle_policy \
  --output artifacts/dgpo_toy/cube_oracle_policy_v1
```

## Baseline failure follow-up: sampler versus denoiser

Read-only cube_sampler_audit.py holds selected continuous4x checkpoint fixed;
evaluates learned and exact conditionalGaussian-mixture mean VP velocities with
DDIM50/100/200 on the same independent4096 continuous conditions x8 candidates.
Analytic velocity is validated against the density-score identity, respects
the existing finite-logSNR schedule, and is never substituted into trained RL.
Same baseline gate, no threshold changes, no new training. This audit can localize
sampler discretization versus learned denoiser fidelity, not determine which
training change repairs the latter. Lowercoverage tests remain gated.

## Continuous-condition revision (user requested)

Run parity_cube_continuous_4x_v1 uses cUniform[-1,1],131072 train samples and
8192 validation/8192 test samples regenerated for the new distribution. Binary
datasets/checkpoints remain unchanged; this is NOT a data-only causal ablation.
For sign-product s=+/-1, mode probability is (0.5-0.4*c*s)/4 in reference,
(0.5+0.4*c*s)/4 in target. Thus all one-/two-coordinate conditional marginals
stay identical, but conditional third moment changes from -0.8c to +0.8c.
At c=0 distributions coincide and oracle reward is zero; difficulty/coverage
varies smoothly with |c|. Global preferred mass averages30% reference versus70%
target, NOT the binary version's10% versus90%. This is not an equally-low-ESS
replication and does not isolate only condition dimensionality.

Reward log[(.5+.4*c*s)/(.5-.4*c*s)] uses nearest-corner sign product, never
provided as denoiser input. Generation uses a uniform midpoint grid of4096
continuous conditions (not used for fitting) with8 independent noises each.
Eight equal condition bins report observed and expected positive-parity mass,
all eight mode probabilities,TV and corner shape. Expected bin probabilities
are averaged over the exact evaluated conditions. Gate requires preferred-mass
absolute error<=.04 in EVERY bin plus the original TV/shape constraints. The
gate is set before observing this run; it is not a relaxation of binary gate.
Data/validation generation seeds unchanged in value but contexts/samples differ
because their distribution changed. Within each variant,4x training includes
its first32k prefix; validation/test do not grow. Batch512, same initialization,
LR,DDIM50,earlystop20epochs/min_delta1e-4.4x data means4x updates per epoch;
this is not a compute-matched data-size study. No binary4x training was launched.

```sh
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.parity_cube \
  --output artifacts/dgpo_toy/parity_cube_continuous_4x_v1 \
  --train-events 131072 --continuous-condition
```

The following sections describe the original binary protocol; all unchanged
training and policy settings also apply to the continuous revision.

New experiment, not a modification of the running12D coverage trial.

Three outputs at eight cube corners (+/-1), independent Gaussian width.15.
Binary condition c=+/-1. A corner is preferred when its sign product equals c.
Reference data assigns total.1 to the four preferred corners, .9 to the other
four, equally within each set. Target reverses these masses. All one- and
two-coordinate conditional marginals are identical between the two mixtures;
only three-way dependence changes. Continuous Gaussian modes have tiny nonzero
overlap; nearest-corner labels are an operational mode diagnostic.

## Pretrain first, inspect actual generation

Fixed32768 train/8192 validation/8192 test samples, disjoint seeded draws.
Randomly initialized conditional neural velocity diffusion,3x128SiLU hidden,
raw noisy3D inputs, scalar binary condition and existing time embedding only.
No parity input, spatial Fourier, fixed transport, Gaussian teacher or EMA.
AdamW1e-3 WD.001, batch512, clipnorm1, ordinary sample-based velocity MSE,
full tUniform[0,1]. Fixed-noise validation, best raw weights. Early stop after20
epochs without accumulated absolute1e-4 improvement; no fixed epoch/step horizon.
DDIM50 is fixed for all sampling/pretrain validation/DGPO in this new toy.

Predeclare generation gate on4096 balanced conditions x8 fresh samples:
each condition preferred mass[.06,.14], eight-mode total variation versus reference
<=.08, max coordinate centroid error<=.12, every per-mode/per-coordinate standard
deviation in[.075,.27], >=90% samples within Euclidean.5 of their nearest corner.
Failure ends the experiment BEFORE RL and is labeled baseline_gate_failed, not
DGPO failure. Gate only the validation-MSE-selected model, not best sampled gate.

## Oracle reward short test

Fixed reward +log9 for preferred corners, -log9 otherwise. This is a categorical
ideal reference-to-target log ratio, NOT exact continuous neural-policy ratio.
Its purpose is to remove classifier error when testing probability response.
No claim of truth convergence from this reward plus a velocity-MSE surrogate.
Pure reward favors100% preferred mass, not target90%; retain all eight probabilities
and mode shape to distinguish utility exploitation from distribution alignment.

Only after gate: two300-update arms from identical best pretrain weights,
fresh AdamW1e-4 WD.001, batch64,K8,M4,tUniform[0,.7], clipnorm1, rawnoEMA.
Native production-derived DGPO kernels; no objective substitution.
Arms: no reference penalty and velocityMSE coefficient1. Reference remains frozen
at baseline. Same rollout seeds across arms; no classifier/refit. Monitor every50
updates plusfirst/final. New independent common-noise endpoint seed660017.

Primary: preferred-mode mass increase relative to initial with paired95%CI
over condition groups; compare arms with paired difference too. A response is
detected if lowerCI>0; call it practically substantial only if gain>=.05 (5pp).
Mode TV versus target and per-mode centroid/width/corner proximity are safeguards,
not interchangeable with reward. Large reward without preserved mode shape is
not successful probability-only transport. Single training seed, short horizon.
If both fail, supervised target fine-tuning is a separate, not-yet-authorized
capacity control; do not automatically add it or alter gates after seeing results.

Offline W&B plus flushed JSONL; no upload. Dataset,bestpretrain,pretrainstate,
baseline and policy states retained. No automatic retry or long-policy extension.

```sh
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.parity_cube \
  --output artifacts/dgpo_toy/parity_cube_v1
```
