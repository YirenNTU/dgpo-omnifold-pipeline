# Pure-reward narrow-joint toy — protocol v1

Declared before running this experiment. This replaces the *research question*,
not the earlier Gaussian/KL results. The user explicitly wants no additive KL,
concentrated ratio weights, and difficulty accumulating fixed reward itself.

## Distribution and minimal model

Let c be uniform on the unit circle. A one-parameter analytic Gaussian
v-predictor generates residual z with deterministic DDIM20, exactly as in the
first toy. Its actual initial endpoint variance is v0. Observe

`x=c; y=(c + Phi(z/sqrt(v0)) - 1/2) modulo 1`.

Both x and y are exactly uniform under **every** residual distribution. At the
initial policy z~N(0,v0), they are independent. Truth's residual distribution is

`p_tau(z) = 0.2*N(0,v0) + 0.8*N(0,tau^2*v0)`.

Its joint density has a smooth diagonal ridge; changing tau changes ridge width,
not either marginal. This is a copula/conditional-residual toy, not H4 physics.
Context is analytically marginalized in training; the model is handed a useful
residual coordinate and need not learn that coordinate from a neural backbone.

Reward is frozen `r=log(p_tau/q0)`, with no learned classifier, clipping,
tempering, reward normalization or refit. Broad tau=0.25, narrow tau=0.0002.
Widths are fixed before training; narrow width targets the *order of magnitude*
of the observed low ESS, not a training failure or exact production curve.

Initial population ratio ESS fraction is exactly

`1 / [1-m^2 + m^2/(tau*sqrt(2-tau^2))]`, with m=0.8.

There are no hard support holes. The reward has a smooth finite maximum at z=0.
A narrow Gaussian policy can approach that maximum. The model does not need
to represent the truth mixture to maximize pure reward, and reward maximization
is **not** claimed to recover truth. Model expressibility is checked by a
finite-DDIM inverse giving policy sigma=0.5*spike sigma (see the instrument
correction below).

## Matched arms

1. `dgpo`: unchanged production LOO advantage and nonlinear detached-gate loss;
   K8, M8 independently gated time/noise draws, uniform t in [0,0.7], shared
   time/noise within candidate groups, detached rollouts, velocity MSE.
2. `score_mc`: same detached candidates and LOO, but exact endpoint log-density
   score-function estimator of negative expected reward. A different estimator,
   explicitly not a production-equivalent DGPO objective.
3. `population`: differentiable one-dimensional numerical quadrature of negative
   expected reward. Same model/optimizer, privileged integration control, **not**
   a compute-matched stochastic estimator.

No additive penalty of any kind. The frozen denoiser still defines the original
DGPO gate's reference MSE, which is *not* an additive reference-trust loss.
AdamW LR0.03, decay0.001, clipping1; one scalar log nominal variance. Toy LR is
not equivalent to a production LR. All arms start at theta0, using common random
numbers; seeds17/29/43. Default600 updates, batch256 groups, DDIM20. Do not choose
best checkpoints or stop early; stop on nonfinite values and mark invalid.

## Measurements and fixed decision rule

Primary: population expected fixed reward at600 and normalized gain
`(R600-R0)/(r_max-R0)`. Also report the full trajectory and update300. Quadrature
uses a scale-adapted integration coordinate so it cannot miss the narrow ridge;
verify values and gradients at doubled resolution before interpreting results.
Independent MC evaluation samples (23927) are separate from training RNG.

Log: reward mean/std; hit probability |z|<3*tau*sqrt(v0); fraction of K8 groups
with a hit and with reward range>1e-3; LOO advantage magnitude; gate; gradient
sign/scale versus population reward gradient; per-group **absolute gradient**
contribution ESS/top1% mass; actual update and resulting population reward gain.
Per-group analytic gradient contributions must sum to the autograd gradient.

Separate global exp(r)-weight ESS, mean within-K ESS, and gradient contribution
ESS. After the policy moves, exp(r) remains p/q0, **not** p/q_current; call its
ESS fixed-ratio-weight concentration rather than current importance-sampling
validity. All-zero gradients have undefined concentration, not ESS=1.

Per seed, call the desired pure-reward slowdown reproduced only if:

- Narrow initial population ESS/N <=1e-3; broad >=0.1.
- Population control gain >=0.5 in **both** widths, validating time budget.
- Broad DGPO gain >=0.5.
- Narrow DGPO gain <=0.25 times broad DGPO gain, and <=0.25 times its own
  population-control gain.

Require all3 seeds for replication; otherwise report mixed/not reproduced, or
inconclusive if population controls fail. This is a **relative learning
slowdown** gate, not proof of a mathematical stationary plateau. The same-width
score-function arm distinguishes a DGPO-specific deficit from a shared finite
sample bottleneck. AUC, truth closure and ESS increases do not replace reward.

If the screen reproduces the slowdown, one predeclared robustness check uses
batch8192 with seed17 and the same600 updates/optimizer. It tests whether the
result was merely a small-batch artifact; it is not compute matched. This is
important because historical production-sized gradients were reproducible.
Do not keep narrowing the ridge or shrinking the batch until failure appears.

## Local execution

```bash
python -m pytest -q experiments/dgpo_toy/test_narrow.py
python experiments/dgpo_toy/narrow.py --seeds 17 --output artifacts/dgpo_toy/narrow_smoke_v1
python experiments/dgpo_toy/narrow.py --output artifacts/dgpo_toy/narrow_v1
```

Only local files under the selected output are written. No NERSC, GPU, Ray,
W&B, checkpoint or dataset. Earlier production changes remain untouched.

## Instrument correction before narrow training

The first smoke completed the three broad arms but stopped **before any narrow
training**: the originally requested witness sigma=0.1*spike sigma was below
DDIM's finite-endpoint noise floor. This is a representability-check failure,
not a pure-reward learning result. Keep the same sampler and widths; use
witness sigma=0.5*spike sigma instead. The primary gain definition, training
budget, optimizer and all reproduction thresholds remain unchanged. Quadrature
still checks 0.1*spike sigma as a mathematical integration test, not as a
claim that the sampler can reach it. The old attempt is recorded separately.

## Exploratory read-only probe declared after the completed screen

The three-seed600-step screen does not pass the predeclared severe-slowdown
threshold: narrow DGPO achieves33.6–37.2% gain, broad96.6%, and narrow score-MC
about99.5%. No threshold changes. No batch8192 training arm is triggered.

To characterize the observed partial deficit, evaluate frozen seed17 DGPO
anchors0/300/600 using four8192-group panels (seeds101/211/307/401), paired
between DGPO and score-MC. Report mean/SE and ratio to the population gradient.
Zero optimizer updates; this is an exploratory follow-up, not a preregistered
training endpoint or a production-gradient reproducibility claim.
