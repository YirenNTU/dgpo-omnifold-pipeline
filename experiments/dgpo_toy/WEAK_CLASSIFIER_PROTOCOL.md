# Saved no-Fourier classifier: matched-policy DGPO ablation

Status update2026-09-21: the endpoint arm below was interrupted on user
direction at1248, with full valid state preserved. User explicitly chose
velocity MSE as proxy; the new matched step0 arm is declared separately in
[PLAIN_VELOCITY_PROTOCOL.md](PLAIN_VELOCITY_PROTOCOL.md). The endpoint run is
not resumed with a different objective and is not a completed10k comparison.

Revised 2026-09-21 BEFORE any execution: user explicitly chose the already
trained no-Fourier classifier. The earlier100-fit-step joint3 draft was NEVER
executed and is superseded. No classifier training in this round.

- Objective: test whether a weaker, higher-ESS learned reward improves DGPO
  transfer and high-order learning, with all policy conditions fixed.
- Success measure: improvement under the SAME original strong H4 judge and
  the known 56 high-order moments, not merely the weak classifier's own score.
- Limiting-factor hypothesis: classifier-to-policy signal actionability.
- First-principles claim: a more diffuse but still informative reward may be
  easier for the unchanged neural DGPO update to absorb.
- Deliverable: one new weak-reward arm versus the completed matched strong arm.
- Budget: ZERO new classifier updates, then 10,000 policy updates, seed17; approximately
  one CPU-hour with direct density, or minutes for pure reward.
- Delete: no new features, classifier fitting/search, calibration/tempering,
  reward rescaling, optimizer reset midrun, refit, two-stage schedule or KL sweep.
- Authority/risk: local toy only; preserve source/control artifacts; no NERSC,
  production or W&B writes. Numerical failure is a validity stop, not success.
- Evidence debt: weak classification does NOT mathematically guarantee high ESS;
  weakness also changes reward scale, approximation error and the implicit
  target. A positive result does NOT isolate ESS causally.

## Saved-classifier intervention (not a pure Fourier/ESS causal ablation)

Use `low_ess_joint3_v1/reward.pt`: original pretrained DDIM20 step0 and frozen
strong joint3 classifier as common judge. Training reward is `plain` from
`conditional_fourier_fit10k_v1/seed17/models.pt`, selected step9900 by validation
BCE among10000 fit updates. Historical held-out AUC.927965/BCE.335261/ESS
fraction.107592. Load its exact weights and saved normalization, freeze all
parameters; no logit temperature/scaling. Inputs retain context and the
existing per-coordinate Gaussian-CDF transform, but no individual or joint
sin/cos features. Hidden widths128,35201 parameters. Classifier checkpoints
contain BITWISE identical pretrained generators (read-only check passed).

The saved plain classifier used a finite training pool; strong joint3 used
fresh data. Feature map, parameter count, fit duration and learned reward
scale/approximation differ. This tests the user-selected classifier bundle,
not only Fourier removal or only ESS. Both see the same truth/generator pair.

Policy always starts at the SAME original pretrained weights, NOT the20k or
no-KL10k endpoint. Preserve seed17, training RNG3017, AdamW1e-4/decay.001,
batch64/K8/M4, DDIM20 and original fixed step0 reference. Only the training classifier
changes. Default `--regularization endpoint` retains direct endpoint coefficient1
and compares to `direct_endpoint_kl1_joint3_10k_v1` (10k, not20k). If the user
chooses pure reward, `--regularization none` compares to the existing
`fixed_reward_joint3_resume10k_v1`. Do not launch both by default. The two-stage
velocity-rescue proposal is NOT part of this round.

## Before-policy measurement gate

Fresh confirmation107017:131072 contexts, one truth/generated pair per context;
measure BOTH classifiers on exactly the same samples. Separate candidate
panel108017:4096 contexts x8, all actual frozen-generator samples.

Require weak AUC>=.70 and less than strong AUC, BCE<=.60,
ESS/N in[.05,.80] and at least10x the strong ESS on the same panel;
within-context centered weak/strong score cosine>.10 and informative
candidate-group fraction>=.01. Saved plain validation BCE/AUC/ESS must reproduce
its selected checkpoint's record to1e-6; configuration and generator weights
must match. Failure means `inconclusive_setup` (or provenance error): save the
outcome, do not silently choose another checkpoint or launch policy.
No policy-selection uses the final endpoint. Existing strong H4 ratio-fidelity
limitations remain explicitly recorded, not retroactively passed.

## Measurements and decision

- Own weak reward/ESS and production-style gradient/head-only concentration,
  gate saturation and clipping every25 policy steps, unchanged training RNG.
- COMMON strong-H4 reward, both reward scales/within-K ESS, 56-moment structure
  and lower-order diagnostics every250. Global ratio ESS is NOT gradient ESS.
- Direct density: same settings and pre/post-update inverse certificate as the
  strong endpoint arm; independent autodiff spot checks every250; fail closed
  and preserve last valid full model/AdamW/RNG/controller snapshot.
- Independent final109017:4096 contexts x8 common noises across step0, saved
  strong-policy10k, and weak-policy10k. Pair by context;500 paired bootstrap
  replicates for high-order MSE changes. Single training seed limitation.

Transfer endpoint: common strong-H4 reward gain>=.10 with lower95% bound>0,
AND weak-policy minus strong-policy common-H4 gain lower95% bound>0.
High-order endpoint:56-moment RMSE at least10% below step0, MSE-change upper95%
bound<0 versus step0 AND versus strong-policy control.
For the endpoint-KL arm, also require existing lower-order limits: max residual
mean<=.10, variance error<=.15, pair covariance<=.08. For pure reward, lower-order
drift remains diagnostic and does NOT veto a successful reward-transfer control.

Passing transfer supports a classifier-signal intervention effect. Passing
transfer + high-order (+ lower-order for KL) supports useful structural progress
in this toy. Neither establishes ESS alone as the mechanism, exact truth
alignment, fresh-classifier closure, or production generalization. A negative
result rules out THIS weak-reward intervention as sufficient at10k only.
No plateau-based early stopping, no peak checkpoint selection. Setup/numerical
failure is unresolved, not a negative algorithmic endpoint.

## Launch

```bash
python -m experiments.dgpo_toy.weak_classifier \
  --source artifacts/dgpo_toy/low_ess_joint3_v1/reward.pt \
  --plain-checkpoint artifacts/dgpo_toy/conditional_fourier_fit10k_v1/seed17/models.pt \
  --comparator artifacts/dgpo_toy/direct_endpoint_kl1_joint3_10k_v1 \
  --output artifacts/dgpo_toy/plain_classifier_endpoint_kl1_10k_v1 \
  --regularization endpoint --policy-steps 10000
```

Local only, no W&B writes. Full state saved after first update and every1000;
progress flushed every25, common strong judge/structure every250. Training
outcome remains unknown at declaration.

## Launch verification (2026-09-21)

Outcome: implemented and launched the requested saved-plain arm; one local
10k trajectory active, not completed.85 tests passed in3.61s. Classifier and
source/control artifacts preserved. No classifier fit or reward rescaling.

Evidence: exact generator match and original validation replay; independent
131072-context panel107017 confirms plain AUC.925025/BCE.340817/ESS.104130,
strong AUC.947359/BCE.207153/ESS.00136307. All declared setup gates pass.
Candidate panel108017 instead has within-K ESS.348643(plain)/.805236(strong),
centered reward RMS1.64359/.73131, score cosine.299263. Do not conflate global
ratio ESS with candidate/gradient concentration. First valid policy update and
full checkpoint saved. No training-success statement follows from setup.

Decision: supports the informative, high-global-ESS contrast; DGPO outcome
unresolved. Learning index for policy effectiveness is not yet available:
no policy hypothesis has been resolved at launch; expected compute~1 CPU-hour.
Deleted: classifier refitting,100-step undertraining, feature engineering,
extra policy arms, two-stage schedules and production/W&B changes.
Next limiting factor: whether this unchanged KL-DGPO update transfers the
plain signal into common-judge reward and true higher-order progress.
Better next round: retain matched final panels and separate global/within-K/
gradient ESS before attributing any outcome specifically to ESS.
