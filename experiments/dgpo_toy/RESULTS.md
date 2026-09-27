# Local DGPO failure-mode toy — 2026-09-21

## Outcome

**Reproduced the endpoint-KL version's initial improvement followed by a wrong
distribution plateau in all three seeds.** This is a sufficient mechanism in
an oracle Gaussian toy, not a causal diagnosis of the production H4 run.

The research workflow removed learned-classifier error, neural capacity,
conditional context, Fourier features, refitting and external infrastructure.
The project research record supplied the question: a local useful direction is
already established; why does improvement not accumulate? This is not another
production one-step sign test.

Protocol was declared before measurement in [README.md](README.md).
Runner: [run.py](run.py). No-update gradient probe: [probe.py](probe.py).
Only local experiment files were added; no production loss or training setup
was changed.

## Minimal system and fidelity

Two trainable parameters control a Gaussian v-predictor with deterministic
DDIM20. In physical coordinates, initial and target distributions have identical
one-dimensional marginals; correlation changes from 0 to 0.8. Rotation into a
fixed 45-degree eigenbasis makes every density and endpoint covariance exactly
computable. The model can represent the target to floating-point precision.

The actual production LOO-advantage and detached-gate DGPO functions are loaded
unchanged from source. The toy retains K=8, M=8 independent time/noise draws,
shared time/noise within candidate groups, uniform t in [0,0.7], mean coordinate
velocity MSE, AdamW, weight decay0.001 and gradient clipping1. LR0.01 belongs to
this two-parameter coordinate system, not production.

The finite-step DDIM distribution is used for reward and exact endpoint KL;
it is not silently replaced by the denoiser's nominal Gaussian covariance.
Tests compare the collapsed linear chain against both production DDIM modes.
Frozen reward is the exact log density ratio, without clipping or tempering.

## Round 1: does the surrogate reproduce non-accumulation?

600 updates per arm; seeds17/29/43; batch256 candidate groups. Initial exact
KL(q||truth)=1.266952. Primary endpoint is update600, not the best selected step.

| Arm | Final exact KL(q||truth), seed range | Final correlation | Independent current Bayes AUC, seed range |
|---|---:|---:|---:|
| Exact negative reward + exact endpoint KL | 0.000000621 | 0.79971 | 0.4981–0.5030 |
| Repo DGPO main + exact endpoint KL, coefficient1 | 0.71744–0.72445 | 0.18696–0.18963 | 0.7175–0.7215 |
| Repo DGPO main, no additive KL | 0.81747–0.82462 | 0.95930–0.95985 | 0.7150–0.7154 |

The analytic control's training trajectory is deterministic, so its three
identical endpoints are not three independent optimizer replications. Its MC
evaluation samples differ by seed. The DGPO arms have independent training
streams and share common random numbers across paired arms.

All seeds pass the predeclared reproduction gate: positive control closes to
within1% of initial KL; first10 DGPO+KL updates improve by at least0.1%; late
mean remains above10%; late-window progress is below1% of initial KL.
The default decision is `replicated_toy_failure`.

Without KL, reward keeps increasing but the distribution overshoots truth:
seed-mean final reward1.807, versus0.510 at the converged exact control; marginal
variance drifts to about3.66 times truth. This arm does **not** reproduce a
flat no-KL reward curve, nor should maximizing frozen reward alone recover truth.

Data: [baseline report](../../artifacts/dgpo_toy/baseline_v1/report.json),
[per-update measurements](../../artifacts/dgpo_toy/baseline_v1/history.csv),
[six diagnostic curves](../../artifacts/dgpo_toy/baseline_v1/curves.png).

## Round 2: can one scalar repair the mismatch?

Before this test, the protocol specified four independent gradient panels,
8192 groups each, at the **initial policy only**. Least-squares matching to the
exact negative-reward gradient gives external main multiplier7.4901616469.
Freeze this number for all subsequent updates; keep the internal gate unchanged.
This changes the toy objective and is not a recommended production coefficient.

| DGPO+KL variant | Final exact KL(q||truth), seed range | Final correlation |
|---|---:|---:|
| Unscaled | 0.71744–0.72445 | 0.18696–0.18963 |
| Initial-gradient scale calibrated once | 0.07979–0.08404 | 0.66450–0.66628 |

This removes about89% of the baseline residual KL. The scaled arm **no longer
passes the original large-residual failure gate** (`not_reproduced`), because
its late mean falls below10% of initial KL. Do not relabel that gate after seeing
the result. A smaller, clearly measured residual remains versus the oracle.

Data: [scaled report](../../artifacts/dgpo_toy/initial_scale_v1/report.json),
[scaled curves](../../artifacts/dgpo_toy/initial_scale_v1/curves.png).

## Why this is not just insufficient AdamW training

For frozen r=log(p/q_ref), the ideal functional obeys exactly:

`-E_q[r] + KL(q||q_ref) = KL(q||p)`.

Thus at the representable truth, the exact composite gradient is zero. The
production main surrogate is not the exact negative-reward gradient. The
following no-update probe evaluates that difference at the truth, independently
of optimizer state or training length:

| Gradient in the two log-variance coordinates | First coordinate | Second coordinate |
|---|---:|---:|
| Exact negative expected reward | -0.392541 | +0.421564 |
| Exact endpoint KL | +0.392541 | -0.421564 |
| Repo DGPO main, four-panel mean | -0.047351 | +0.022568 |
| Main Monte Carlo standard error | 0.000092 | 0.000218 |
| Repo main + exact KL | +0.345190 | -0.398996 |

The DGPO-main component perpendicular to the KL gradient is
-0.019275 ±0.000097 (mean ± Monte Carlo SE across four panels, not a production
confidence interval). All four panels have the same sign. Even the best positive
scalar **at truth**, used solely as an optimistic diagnostic, leaves composite
gradient norm0.21166. A single scalar does not make this toy's truth stationary.

At the seed17 stalled endpoint, mean composite gradient is approximately
(-0.00092,-0.00266), whereas the exact KL-to-truth gradient is approximately
(-0.20633,+1.31152). The surrogate can therefore nearly balance the penalty at
the wrong distribution. Near-opposing main/KL gradients alone are not a bug:
the exact control also has cancellation, but at the correct distribution.

The mean gate is about0.46 at truth and0.48 near the stalled distribution;
wholesale sigmoid saturation is not the explanation in this toy.

Data: [gradient probe](../../artifacts/dgpo_toy/gradient_units_v1.json).

## Boundaries of the conclusion

- This is a two-variable correlation task, **not irreducible higher-order H4**.
- Full Gaussian support removes hard support holes, not every finite-sample
  tail effect. There is no real-data narrow-angle coverage construction here.
- Analytic Gaussian denoisers, finite DDIM, time weighting, the velocity-MSE
  proxy and the nonlinear gate are still part of the tested mechanism. We have
  not isolated which of these causes the remaining directional distortion.
- Exact endpoint KL removes learned-KL-critic error. Production uses estimated
  critics, conditional neural networks and different optimization scales.
- The positive control integrates expectations analytically and is not a
  compute-matched stochastic gradient estimator.
- The experiment shows a candidate sufficient mechanism. It does not establish
  that production coverage, classifier calibration, or optimizer noise is irrelevant.
- Fixed reward standard deviation need not vanish at truth: it is about0.801
  for the converged exact control. Reward/std alone is not a closure criterion.

## Fast local loop

From the repo root:

```bash
/opt/miniconda3/envs/MyEve/bin/python -m pytest -q experiments/dgpo_toy/test_toy.py
/opt/miniconda3/envs/MyEve/bin/python experiments/dgpo_toy/run.py --output artifacts/dgpo_toy/baseline_v1
/opt/miniconda3/envs/MyEve/bin/python experiments/dgpo_toy/probe.py --baseline-report artifacts/dgpo_toy/baseline_v1/report.json --output artifacts/dgpo_toy/gradient_units_v1.json
/opt/miniconda3/envs/MyEve/bin/python experiments/dgpo_toy/run.py --main-scale 7.4901616469227665 --output artifacts/dgpo_toy/initial_scale_v1
```

Replace the Python path for another environment with torch, numpy, matplotlib
and pytest. Commands need no W&B, Ray, dataset, checkpoint, network or GPU.
In the Codex sandbox PyTorch shared-memory initialization required an approved
unsandboxed local process; no external job was submitted.

Nine validation tests passed, covering density identities, representability,
DDIM parity/derivatives, exact production kernel, reward moments, deterministic
training, evaluation independence, external-scale isolation, and an inconclusive
decision if the positive control fails. Reports include source SHA256 hashes.

## Research round close

- **Outcome/evidence:** all-three-seed baseline reproduction, substantial benefit
  from initial scalar calibration, remaining nonzero vector field at truth.
- **Decision:** keep this toy as a fast test bed; do not change production yet.
- **Deleted:** learned classifier, Fourier, context, refit, EMA, real-data loading,
  learned KL critic, Ray, W&B and GPU execution for this first question.
- **Learning index:** final baseline7.45s + scaled7.43s + probe0.61s =15.49s measured
  workload for two hypothesis updates, about7.75s/update. This excludes imports,
  plots, engineering time, smoke runs and reruns; it is not total task wall time.
- **Next bottleneck:** whether replacing the velocity-MSE/gate surrogate with an
  exact log-density score-function gradient restores closure using the *same*
  detached samples and LOO baseline. This is a new stochastic-estimator control,
  not another production single-step direction experiment. Not yet executed.
- **Better next round:** retain identical seeds, endpoint and optimizer; change
  only the main estimator. Verify its expected gradient at truth before running
 600 updates, then inspect stationary-point and distribution metrics together.
