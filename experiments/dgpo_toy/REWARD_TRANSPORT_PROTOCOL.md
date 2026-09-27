# Does reward guide mode probability transport? A/B/C diagnostic

Status: implemented; NOT launched by the assistant. Local toy only; no NERSC or
production changes. This is a one-seed mechanism pilot, not a closure result.

## One question

Can the same Fourier-conditioned diffusion, native DGPO and coefficient-1
velocity surrogate learn conditional mode probabilities when the reward cannot
increase through within-mode shape changes?

Earlier evidence: `shape_matched_dgpo_v1` learned correct average mode ordering
(cosine .971), with ESS/N .843, but conditional mode TV barely moved after1000
updates. Removing the reference penalty collapsed the raw-policy endpoint.
The earlier k8 oracle successes are not this nonperiodic learned-reward case.

## Fixed source and arms

- Source: `artifacts/dgpo_toy/nonperiodic_cube_classifier_extended_v1/reference.pt`.
  Trained on the uniform cube reference, NOT full truth. No new pretraining.
- Frozen critic: `artifacts/dgpo_toy/shape_matched_dgpo_v1/classifier.pt`, selected
  step3100 by validation BCE. Reuse, do not refit it.
- A: original learned logit r(c,y).
- B: estimated E_q0[r(c,Y) | c, sign-mode(Y)]. Removes within-mode variation;
  preserves conditional average scores approximately, not the exp-score mean.
- C: log p_truth(mode|c) - log q_hat_initial(mode|c). Denominator is sampled from
  the actual starting diffusion, never the nominal uniform distribution.
- All three start at identical velocity/sample functions and identical weights.
  Same zero-initialized Fourier adapter k=[1,2,4,8]; no raw-basis extra arms.
- Same native DGPO loss, nonlinear detached gate and unscaled leave-one-out
  advantage. AdamW1e-4, WD.001, clip1, batch64, K8, four t~U[0,.7], DDIM50,
  policy seed17,1000 updates, fresh optimizer per arm, frozen starting reference.
  The penalty is coefficient1 times HALF velocity MSE, not exact endpoint KL.
  No reward centering beyond native LOO, standardization, clipping, temperature,
  extra features, EMA, pathwise gradients, or new optimizer objective.

Only reward changes; its change is explicitly authorized. B/C do change the
reward objective. They do NOT constitute a target-preserving variance reduction.
They use known toy mode labels for diagnosis, not as a production proposal.

## Continuous-condition calibration

Generate2048 samples at each of257 nodes including c=-1,+1. Estimate all eight
mode probabilities and within-mode mean logits. Two disjoint1024-sample halves
test calibration reliability. Every half/node/mode must have>=32 samples.
No empty-cell fill, cross-condition donor mixing, or uniform fallback.

B linearly interpolates mean scores; C linearly interpolates q (not log q), then
uses the analytic nonperiodic truth numerator at the actual continuous c.
Thus the model still trains on continuous conditions, not bin centers. The table
is a finite-sample/interpolation approximation, NOT an exact neural density.
Gaussian orthant crossings in the toy truth have negligible but nonzero mass;
the categorical numerator refers to the designed corner mixture.

An independent midpoint grid256 x1024 samples tests interpolation and support.
Predeclared validity gates, before any policy update:

1. Split-half centered score cosine>=.90 for both B and C.
2. B midpoint mean-score RMS error / conditional between-mode RMS<=.25.
3. Midpoint q-probability RMS error<=.025; all midpoint counts>=32.
4. Independent C reweighting reduces conditional mode TV by>=.01.

Validation includes Monte Carlo error. No threshold tuning or reward changes
after observing gates; a failed gate is inconclusive setup and stops training.
Calibration and validation use separate seeds, neither reused for policy/audits.
Reweighting validates the reward table; it is not substituted for DGPO.

## Measurements and interpretation

Endpoint:128 fixed midpoint conditions x1024 candidates, identical latent noise
for source/A/B/C. Evaluate all three frozen rewards on EVERY policy. Never compare
raw A gain numerically with raw B/C gain as equivalent reward scales.

Record full8-mode TV/RMSE, parity error, conditional first/pair sign moments,
near-corner fraction, corner-distance RMS, missing/sparse cell counts, and all
three reward gains. Decompose learned-A gain into mode-at-source-shape and the
remaining shape/interactions. Endpoint missing modes are measured (mean score
unavailable), not grounds to abort remaining arms. A numeric nonfinite failure
is still an error. Persist policy/optimizer/RNG checkpoints BEFORE evaluation.

Every100 policy steps: smaller64x256 structure panel and native gradient traces,
including reward/reference norms, cosine, total-on-reward projection, within-K
ESS, informative groups, gate saturation, clipping and actual update L2.
These monitors are not checkpoint-selection criteria. Last prescribed step is
the endpoint, not the most favorable intermediate point.

Pilot transport flag: endpoint mode TV decreases>=.01, corner fraction drops
no more than.02, and no endpoint mode is missing. This is a descriptive threshold,
not training-seed significance. Report B-A,C-A,C-B paired fixed-panel contrasts.

- B passes/A fails: supports an actionable within-mode-score contribution.
- Only C passes: supports mode-score fidelity/calibration as a contributor;
  ordinal preferences alone were insufficient. Does not uniquely isolate scale.
- All fail: reward errors are not a sufficient explanation at1000steps; remaining
  bottleneck is probability transport under this representation/loss/constraint.
- Other combinations: mixed/control improves; do not force the hypothesis.

## Fresh classifier endpoint (enabled by default)

After all three policies, cold-fit four matched Fourier classifiers against the
ORIGINAL FULL truth (source,A,B,C). Same initialization seed41, paired continuous
conditions/truth and latent noise, independent train32768/val8192/test16384 pools.
AdamW3e-4, WD.001, balanced batch512, clip1. Check each100 updates, minimum2000,
patience20 checks, min_delta1e-4, maximum16000. Joint stopping after every arm has
plateaued; each arm selects its own lowest-validation-BCE checkpoint. No test
data used for weight selection. Fit exhaustion is `audit_inconclusive`, not closure.

Primary distribution endpoint: delta abs(test AUC-.5), baseline contrasted with
A/B/C and matched pairwise arms; held-out BCE is a companion diagnostic.
Report300 paired-context bootstrap intervals conditional on fitted classifiers,
NOT policy/audit-seed uncertainty. A negative upper bound is evidence of reduced
discriminability under this audit, NOT proof of full distributional equality.
Mode transport alone never establishes this endpoint. `--skip-audit` explicitly
marks `completed_without_audit` and prohibits a classifier-closure conclusion.

## Outputs and launch

`report.json`, live `progress.jsonl`, `calibration.pt`, frozen `classifier.pt`,
`{A,B,C}_last.pt` (raw weights, optimizer, RNG, full native history), paired
endpoint tensors, `cold_audit/` panels, selected and last audit weights/scores.
W&B project `dgpo-toy`, group `Conditional reward transport`; offline by default.
Run name: `Can reward move mode probabilities? | Fourier cube | A/B/C reward | V MSE 1`.
Policy/structure/audit metric namespaces each have their OWN arm-specific step
axis. The logging event counter is not a policy training step.

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.reward_transport \
  --output artifacts/dgpo_toy/reward_transport_abc_v1 \
  --steps 1000 --wandb-mode offline
```

Add `--preflight` for read-only lineage/config validation without creating output,
initializing W&B, generating panels, or training. Output must be a fresh directory;
the preceding experiment artifacts are never overwritten. No automatic job
submission or remote synchronization. Toy upload exclusions remain unchanged.
