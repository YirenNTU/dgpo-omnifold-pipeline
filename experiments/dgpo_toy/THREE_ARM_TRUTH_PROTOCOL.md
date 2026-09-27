# Truth-trained diffusion: same-data weak/strong rewards, three DGPO arms

2026-09-21. User authorizes ONE new weak classifier and exactly THREE policy
experiments. Primary endpoint is OWN fixed-reward absorption, NOT high-order
moment improvement or fresh-classifier closure. Secondary metrics cannot veto it.

## Inputs and one new classifier

- Initial/reference: truth_diffusion_earlystop_v1/best_model.pt, RAW epoch25/1600.
- Truth: truth_dataset_seed17_v1/dataset.pt,32768train/8192val/16384test rows.
- Strong: truth_diffusion_matched_h4_v1/best_classifier.pt, joint3,selected4/512.
- Weak: fit plain/no-Fourier from scratch with the SAME saved panels.pt from
  strong H4, including generated negatives. No additional truth OR negatives.
  Same hidden128,AdamW3e-4/decay.001,batch512,train-only standardization50,
  validation-BCE patience20/min_delta1e-4; raw best selected, no forced weak AUC
  or high ESS gate. Keep the existing per-coordinate CDF transformation.

The finite-pool regime is now matched. Features, parameter count, fitted reward
scale/ranking and selected fit duration still differ. Measure the observed
weakness/ESS contrast; do not claim it was achieved merely by choosing plain.
Reproduce selected validation BCE from the saved weights before policy training.

## Three arms, all from the SAME initial raw model

|Arm|Frozen reward|Additive velocity coefficient|
|---|---|---:|
|strong_velocity|Strong joint3|1|
|weak_velocity|Plain/no-Fourier|1|
|strong_no_kl|Strong joint3|0|

10000 policy updates EACH,seed17/train RNG3017,AdamW1e-4/decay.001,
batch64/K8/M4,DDIM20,t uniform[0,.7],clip1. No teacher, refit, EMA, logit scaling,
tempering, classifier gradient through DDIM or direct endpoint-KL controller.
Use unchanged production DGPO LOO-unscaled/detached gate and same original
step0 reference. "No KL" removes only the additive penalty, not the gate reference.
Velocity penalty = coefficient*.5*mean((v_current-v_step0)^2), same detached
candidates/t/noise as the main loss. Proxy, NOT exact endpoint KL.

Policy contexts/noises are freshly drawn as in the prior toy; no additional
truth labels are consumed. All arms use the same random stream; do not pass
one arm's policy/optimizer into another. Each starts fresh AdamW from step0.
No early stop on reward/structure, no peak checkpoint selection; stop on
nonfinite numerical failure and preserve last complete saved state.

## Measurements, decisions and limitations

Every25 updates: own reward/gain/paired CI/std, global ratio ESS, within-K ESS,
head-gradient concentration, gate saturation/clipping/update norm. Velocity
arms also record actual MSE/weighted penalty and pre-Adam component norms,
cosine, total-on-main projection. These batch diagnostics do not prove the
population-gradient or optimizer mechanism.

Every250: same fixed strong and weak judges, true56 moments/marginal diagnostics.
No classifier training at audit points. Monitor seed130017; independent final
seed132017,4096 contexts x8, paired by condition with common initial noise.
No fresh truth sampling is needed to score reward; known moments are evaluation-only.

PRIMARY per arm: final-minus-initial OWN reward, raw units and divided by that
judge's fixed initial generated-reward SD. Reliable gain = lower95%CI>0.
Operational nontrivial gain = at least.01 initial-SD with lower95%CI>0;
report both so a smaller genuine gain is not called "no learning".
Normalization is reporting only; NEVER modify training reward/penalty balance.

PRIMARY contrasts:
1. Strong no-KL minus strong velocity, same strong judge: positive paired95%CI
   supports suppression by coefficient1 proxy at this budget/seed.
2. Weak-velocity own gain/weak initialSD minus strong-velocity own gain/strong
   initialSD: positive paired95%CI supports better absorption of the weak
   reward bundle. It is not an ESS-only or reward-scale-controlled causal test.

Common strong reward and truth structure are SECONDARY. No-KL marginal drift
does not veto reward absorption. Lower ESS itself does not establish gradient
concentration. Strong H4 already has poor ratio normalization; explicitly retain
that limitation rather than treating logits as an exact truth ratio.

Outcome possibilities: penalty contrast only -> proxy effect; weak contrast
only -> classifier-bundle actionability effect; both -> both contribute in this
toy; neither -> these interventions insufficient at this budget. There is no
weak-no-KL fourth arm, so the complete classifier x penalty interaction remains
unidentified. Single training seed; panel CIs do not quantify seed variance.

## Scope and budget

Objective/deliverable: same-data weak classifier plus three10k full-state policy
trajectories with independent endpoint. Limiting factor: reward absorption under
the proxy. First-principles claim: matched source/RNG/objective isolates proxy
presence within strong H4, while the second contrast tests the reward bundle.
Time: approximately minutes per local CPU arm; no W&B/production/NERSC writes.
Delete: new features, reward refits, optimizer sweeps, extra seeds and fourth arm.
Risk/authority: preserve every old artifact; new output directories only.
Evidence debt: feature/scale/prior differences, finite datasets, imperfect ratios.

Weak fit command:

```bash
python -u -m experiments.dgpo_toy.matched_h4 \
  --dataset artifacts/dgpo_toy/truth_dataset_seed17_v1/dataset.pt \
  --diffusion-checkpoint artifacts/dgpo_toy/truth_diffusion_earlystop_v1/best_model.pt \
  --panels-from artifacts/dgpo_toy/truth_diffusion_matched_h4_v1/panels.pt \
  --feature-mode plain \
  --output artifacts/dgpo_toy/truth_diffusion_matched_plain_v1 \
  --patience 20 --min-delta 0.0001
```

Three-arm launch (run from repository root; no classifier refit):

```bash
python -u -m experiments.dgpo_toy.three_arm_truth \
  --dataset-path artifacts/dgpo_toy/truth_dataset_seed17_v1/dataset.pt \
  --diffusion-path artifacts/dgpo_toy/truth_diffusion_earlystop_v1/best_model.pt \
  --strong-path artifacts/dgpo_toy/truth_diffusion_matched_h4_v1/best_classifier.pt \
  --weak-path artifacts/dgpo_toy/truth_diffusion_matched_plain_v1/best_classifier.pt \
  --output artifacts/dgpo_toy/truth_three_arm_v1 --steps 10000
```

Implementation verification: 109 toy tests pass. The weak fit selected epoch 62
(stopped at 82), test BCE 0.439938, AUC 0.868737, ESS fraction 0.203027.
Strong selected test BCE 0.425024, AUC 0.899245, ESS fraction 0.00053405.
This verifies an informative high-/low-ESS contrast, not an ESS-only intervention.
Full optimizer/RNG checkpoints are saved at step 1, every 1,000, and the endpoint;
the runner does not currently expose an automatic resume CLI.
