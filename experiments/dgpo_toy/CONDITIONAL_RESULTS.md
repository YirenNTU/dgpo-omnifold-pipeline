# Conditional neural diffusion / Fourier reward — local results, 2026-09-21

## Outcome

Implemented and tested the complete experiment, including both policy-update
paths. Two full classifier/pretraining pilots completed locally. Neither passed
the predeclared setup gates, so **no eligible full-budget DGPO or pathwise
trajectory was run**. Smoke updates verify execution only, not reward transfer.
Decision: `inconclusive_setup`, not a reproduced DGPO failure.

The workflow's instrument-first rule prevented choosing an overfit low-ESS
checkpoint merely to satisfy the desired failure scenario. The research-memory
rule kept held-out discrimination, weight concentration, and policy improvement
as separate claims. No production files, W&B runs or NERSC jobs changed.

## This is a conditional, many-parameter diffusion

12 generated coordinates, three context inputs, a 10,444-parameter MLP v-model,
and an actual 20-step DDIM chain. Baseline pretraining used 5,000 updates of
supervised analytic-Gaussian v-teacher distillation. The neural forward pass
does not call that teacher. It is not the previous analytic scalar generator,
but also does not reproduce EveNet's entire noisy-target pretraining history.

Truth has four context-dependent three-variable phase relationships. Its exact
conditional univariate and bivariate marginals equal the Gaussian baseline's;
the model must change joint dependence, not simply fix a marginal variance.
The classifier sees individual marginal-CDF coordinates, context, and optional
individual harmonics1..4. It is NOT given the triple groups or joint formula.
This known marginal transform is a synthetic coordinate choice, not a claim of
physics-prior-free equivalence to H4's engineered pair-angle features.

The learned baseline passed every initial lower-order check:

| Check | Measured | Predeclared maximum |
|---|---:|---:|
| Maximum residual Gaussian KS | .010705 | .06 |
| Maximum variance error | .023289 | .15 |
| Maximum off-diagonal covariance | .021095 | .08 |
| Maximum context-bin mean error | .043006 | .10 |

Truth-minus-baseline known third-order signal was .847267, above .20.
Thus the pretraining/missing-joint part of this toy is operationally established.

## Classifier results

Seed17; identical generated/truth datasets and training budgets within each
plain/Fourier comparison. Selection uses validation BCE; values below are on
16,384 independent test contexts per class. No test-based checkpoint selection.

| Fit budget | Classifier | Selected update | Test BCE | Test AUC | Test ESS/N | log mean ratio |
|---:|---|---:|---:|---:|---:|---:|
| 5,000 | plain | 400 | .693230 | .500799 | .999135 | .000461 |
| 5,000 | Fourier | 5,000 | .551569 | .822912 | .072888 | .539323 |
| 10,000 | plain | 9,900 | .335261 | .927965 | .107592 | -.316856 |
| 10,000 | Fourier | 6,300 | .522196 | .857131 | .026689 | .862259 |

The10k follow-up was declared AFTER preserving the5k result and BEFORE its run;
only the classifier budgets changed. Both runs use exactly the same baseline.
Hidden widths match, but parameter counts do not: plain35,201 / Fourier47,489.
These are finite-budget architecture comparisons, not proof of Fourier's unique
causal contribution or its necessity.

Fourier exposes held-out signal earlier, but the plain classifier later overtakes
it in BCE/AUC. At the final Fourier10k update, training BCE=.224446 while
validation BCE=.623620, worse than best validation BCE=.527785 at6300. The
final validation ESS/N=.001815 and log mean ratio=3.19883 cannot justify replacing
the best-BCE checkpoint with this overfit checkpoint. More concentrated weights
are not automatically a better density-ratio estimate.

## Why formal policy updates were not run

The selected Fourier reward fails BOTH original gates in both pilots:

- Required ESS/N<=.01; observed .072888 / .026689.
- Required abs(log mean ratio)<=.50; observed .539323 / .862259.

The synthetic nominal oracle has population ESS/N=.002163, but that does NOT
make the learned classifier a low-ESS reward. The oracle denominator is the
nominal conditional Gaussian, not the exact learned-neural endpoint density.
The oracle is diagnostic only and is never substituted for the classifier.

Normalization is a ratio diagnostic, not proof of a reward-gradient failure:
a constant logit offset cancels under LOO and does not alter ESS. We retain the
declared gate and do not use its failure to claim a DGPO bottleneck. The low-ESS
gate itself also failed, independently of normalization.

**Supports:** this conditional neural baseline is suitable for measuring missing
third-order structure, and learned classifiers can discover held-out differences.
**Unresolved:** reward improvement under a valid learned low-ESS setup; low ESS
as the main production bottleneck; production transfer of any toy mechanism.

## Tests and diagnostics

31 tests pass across the three toy families. New tests cover copula moments and
lower-order marginals, analytic ESS quadrature, context dependence, many-parameter
DDIM gradients, standardization preserving logits, production loss detachment,
per-context last-layer gradient sums matching autograd, context-cluster standard
errors, decision validity and evaluation RNG isolation.

Two3-step smoke arms run, including the full differentiable-DDIM pathwise control.
The DGPO head-gradient concentration diagnostic is exact for its final linear
layer ONLY, not a claim of whole-network gradient ESS. Full-policy logging also
records global fixed-weight ESS, within-K ESS, informative groups, gate, clipping,
update norm, reward and known-joint progress. Final endpoints use a separate
context/noise panel; no policy checkpoint is selected on monitor/test reward.

## Artifacts / reproduce

- [5k report](../../artifacts/dgpo_toy/conditional_fourier_v1/seed17/report.json)
- [10k report](../../artifacts/dgpo_toy/conditional_fourier_fit10k_v1/seed17/report.json)
- [updated smoke](../../artifacts/dgpo_toy/conditional_smoke_v2/seed17/report.json)
- Initial generator and selected plain/Fourier classifiers are saved in each
  run's `seed17/models.pt`, with config; live phase histories are `progress.jsonl`.

```bash
python -m pytest -q experiments/dgpo_toy
python experiments/dgpo_toy/conditional.py \
  --output artifacts/dgpo_toy/conditional_fourier_v1 --seeds 17
python experiments/dgpo_toy/conditional.py \
  --output artifacts/dgpo_toy/conditional_fourier_fit10k_v1 \
  --seeds 17 --classifier-steps 10000
```

## Round close

- Outcome: implemented conditional neural/Fourier test; setup incomplete for
  the intended low-ESS policy question. No fabricated failure/reward endpoint.
- Evidence: baseline checks, held-out classifier measurements,31 passing tests.
- Decision: `unresolved` for low-ESS DGPO reward transfer.
- Cost:17.98s+31.89s=49.87s local pilot compute; excludes coding/imports/tests.
  Two useful updates (baseline valid; Fourier representation alone does not
  establish the desired ratio regime) give ~24.94 compute seconds per update.
- Deleted: production edits, infrastructure, refits, KL, logit rescaling, and
  selecting a worse-BCE checkpoint solely for its lower ESS.
- Next limiting factor: a learned reward with independently validated tail
  behavior in the intended low-ESS regime, not another unconditional scalar toy.
- Better next round: separate representation/ratio calibration from the policy
  experiment before attributing stalled reward to diffusion optimization.
