# A moving Gaussian: does the conditioning function change Fourier's MSE benefit?

Status: **prepared; no scientific training launched by the assistant**.
This round concerns supervised velocity MSE only. No classifier, reward,
DGPO, KL, EMA, generation metrics, Ray, remote submission, or cloud sync.

Prior evidence: [the completed k8 cube comparison](../../artifacts/dgpo_toy/cube_truth_fourier_v1/RESULTS.md)
improved held-out velocity MSE in all three seeds (e.g. `.795550 -> .777112`
for seed17). It used a basis containing the known true frequency. This new
round asks whether the benefit survives changing the conditional function;
it does not reinterpret the old result as MSE degradation.

## Intuition

A knob `c` is observed, continuously and uniformly in `[-1,1]`. Turning the
knob moves the center of a Gaussian cloud along one spatial axis:

```
y | c ~ Normal(f(c), 0.35^2)
```

At each knob position there are many possible answers, not one deterministic
regression label. We train an actual conditional diffusion velocity network
from noisy targets and random diffusion times. We do not sample its reverse
chain in this round, because the user has selected velocity MSE as the endpoint.

Three predeclared functions, without tuning Fourier frequencies per function:

| Case | Center motion | Purpose |
| --- | --- | --- |
| linear | `sqrt(3)*c` | Easy, nonperiodic control; extra features are unnecessary in principle. |
| periodic | `sqrt(2)*sin(4*pi*c)` | Positive control: the fixed basis includes the true frequency. |
| bump | A Gaussian bump centered at `.25`, width `.15`, centered/rescaled analytically | A localized, nonperiodic dependence, not exactly a finite Fourier sum. |

All functions have exactly zero mean and unit variance under uniform `c`.
All have the same conditional Gaussian width, hence the same unconditional
target variance and the same irreducible velocity MSE. This matches scale,
not every aspect of distribution shape. A bump may still be approximated
using Fourier features; it is not designed to force a Fourier failure.

## What is held fixed?

- Fixed complete-truth sampled data: 32,768 train / 8,192 validation / 16,384 test.
  Three training seeds `17,23,41` reuse one dataset per function; this is not
  independent-dataset replication. All cases share the underlying `c`/noise draws.
- Raw arm: noisy `x_t`, raw observed `c`, and the old toy's time embedding.
- Fourier arm: identical network plus a **zero-initialized** `8 -> hidden`
  additive condition projection before the first SiLU. Raw `c` remains present.
  Features are `sqrt(2)*sin(k*pi*c), sqrt(2)*cos(k*pi*c)`, fixed `k=1,2,3,4`.
  Raw `c` is scaled by `sqrt(3)` in both arms. Every condition feature therefore
  has population variance one. No train/validation-derived normalization changes.
- Three hidden layers, width64, inherited toy velocity output parameterization
  `sigma(t)*MLP(...)`. Same shared weights and exactly equal velocities at step0.
- Raw arm has a same-size adapter receiving zero features. Stored parameter
  counts match, **active capacity does not**. This is an add-Fourier test, not
  proof that Fourier beats every equal-active-capacity alternative.
- Fresh AdamW in both arms: lr `1e-3`, weight decay `.001`, clip norm1, FP32,
  CPU, no EMA. Constant LR is intentional; no scheduler confound in this round.
- Identical minibatches, time draws, and diffusion-noise draws within each pair.
  Training sees only sampled `(c,y)`; it never calls the truth mean or oracle.

## Objective and instrument

Use the existing toy's finite-logSNR cosine VP schedule, with
`a^2+s^2=1`, and ordinary unweighted sample velocity MSE:

```
x_t = a*y + s*epsilon
v_target = a*epsilon - s*y
loss = mean((v_model(x_t,t,c) - v_target)^2)
```

The analytic conditional optimum provides an evaluation-only instrument.
Writing `tau=.35`, `D=a^2*tau^2+s^2`, and `mu=f(c)`:

```
v_star = -s*mu + a*s*(1-tau^2)/D * (x_t-a*mu)
irreducible_MSE(t) = tau^2/D
```

In expectation, `MSE(model) = irreducible_MSE + E[(model-v_star)^2]`.
Finite-panel sample MSE need not equal that sum exactly because of its random
cross term. `oracle_excess_mse` measures the same velocity estimation error,
not a different physics endpoint and never a training target. The optimum
integrated MSE is approximately `.35` for this near-full cosine time interval.

## Primary endpoint and stop rule

Primary: **test velocity MSE at each arm's best-validation checkpoint**, under
the same available number of optimizer updates. Include step0 as a selection
candidate. Test is evaluated only after stopping, never used for checkpoint
selection, early stopping, or LR changes.

Evaluate fixed, fresh-noise train-probe and validation panels each epoch. Keep
training **both** arms until neither has made a `1e-4` absolute validation
improvement over its patience anchor for20epochs. Small gains accumulate
against that anchor. The exact lowest validation loss selects checkpoints,
independently of the early-stop min_delta. Both arms always get equal updates.
There is no default epoch/step cap. Optional `--max-epochs` is explicitly
reported as `budget_limit_not_convergence` when it stops the fit.

Report `delta = MSE_fourier - MSE_raw` with an event-paired approximate95%
interval on the independent test panel. Predeclare material difference `.005`:

- Upper bound below `-.005`: supports Fourier benefit for that case/seed.
- Lower bound above `+.005`: reproduces material Fourier degradation.
- Otherwise: unresolved; **not proof of equivalence**.

Require the same decision in all three training seeds before describing a
replicated result. Per-seed intervals are pointwise, conditional on the models,
not simultaneous intervals across cases and not training-seed uncertainty.
Plateau here means the declared stopping rule, not proof of global optimality.

## Diagnostics that can change the next decision

Only velocity-error diagnostics: online train MSE; fixed train-probe MSE;
validation MSE; evaluation-only oracle excess; MSE in ten time/condition bins.
Log clip fraction and max gradient norm to flag an invalid optimization regime.

| Observation | Next bounded question, not an automatic extra run |
| --- | --- |
| Fourier helps periodic but not linear/bump | Is the gain basis/task alignment, rather than universally better conditioning? |
| Fourier worsens both train-probe and validation MSE | At the same initialization, does a matched lower-LR comparison remove the optimization failure? |
| Fourier lowers train-probe MSE but raises test MSE | Does more independent training data remove the generalization gap? |
| Both stall well above the analytic optimum | Which time/condition slice accounts for the excess, before changing architecture? |
| Fourier helps all three | This toy does **not** reproduce the real loss degradation; do not manufacture a failure or attribute EveNet's gap to it. |

This toy changes the conditional mean, not a high-order copula. That deletion
is intentional for velocity-MSE diagnosis. It cannot establish EveNet's unique
bottleneck, an attention-specific mechanism, or high-dimensional closure.

## Commands (local only; user starts training)

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
/opt/miniconda3/envs/MyEve/bin/python -m experiments.dgpo_toy.conditioning_mse \
  --output artifacts/dgpo_toy/conditioning_mse_v1
```

Default: all three functions and all three seeds, raw/Fourier paired per fit.
For a pilot only, add `--seeds 17` and use a separate output path. Do not call
one seed decisive. Previewing the design launches no training:

```bash
/opt/miniconda3/envs/MyEve/bin/python -m experiments.dgpo_toy.conditioning_mse \
  --output artifacts/dgpo_toy/conditioning_mse_design --preview-only
```

Artifacts: `design.png` (truth design, not generated results), `mse_curves.png`
(train-probe/validation curves), `progress.jsonl`, `report.json`, fixed datasets,
and each pair's selected/last raw checkpoints. There is no automatic resume;
use a new output for a new experiment. Partial progress is still in JSONL.
All files stay under toy paths already excluded from NERSC synchronization.

## Round record

- Outcome: prepared a minimal, testable conditioning-function intervention.
- Evidence: 21 contract tests passed in2.38seconds; they validate scale matching,
  oracle MSE, paired noise, initial equality, no teacher leakage, stopping,
  checkpoint selection, and runner pilot/budget labels. Design preview rendered
  and visually inspected. No scientific fit executed.
- Scientific decision: **unresolved; training has not been launched**.
- Learning index: pending measured training cost / changed hypothesis states;
  unit tests are implementation evidence, not a scientific result.
- Deleted: classifier, ESS, DGPO, KL, sampling, physics metrics, remote jobs.
- Next limiting factor: whether Fourier's MSE effect depends on the true
  conditional function, under a matched optimization protocol.
- Better next round: act on the observed MSE pattern, not presumed Fourier harm.
