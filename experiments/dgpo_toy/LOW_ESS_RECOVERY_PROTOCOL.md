# Learned low-ESS reward recovery — v1, declared before execution

## Question and single intervention

The previous12D conditional neural toy's best-validation Fourier classifier did
not reach ESS/N<=.01. Continuing the finite32,768-event training pool to10k
overfit: train BCE decreased while validation BCE worsened and logits inflated.
Do not select that inflated checkpoint or weaken the original low-ESS gate.

Freeze the saved initial neural DDIM20 generator, distribution, kappa8,
structured mass.9,12 dimensions, individual Fourier harmonics1..4, MLP sizes,
AdamW, LR, and training-only encoder-output standardization. The ONE training
intervention is replacing reused classifier fitting examples with fresh paired
truth/ACTUAL generator samples each update. No analytic-generator substitution.

Fit the Fourier classifier from the same initialization for a bounded20,000
updates (the10k prefix also provides a budget-aligned comparison to the prior
finite-pool run). Only validation BCE selects its checkpoint. No test-based
selection, oracle labels/targets, logit rescaling, temperature, clipping, refit,
new features or known triple identities. The known density is diagnostic ONLY.
Keep the previous pilots intact. No automatic budget extension beyond20k.

The existing training panel is reused only to keep the original standardization
statistic source fixed. Validation retains the original independent8192 contexts;
all new fit contexts/noises have a separate RNG. Confirmation now uses131,072
NEW contexts per class, not the prior test panel, to resolve ratio tails better.

## Check the nominal denominator rather than assume it

The saved generator is a neural finite-DDIM map F_c(z), not exactly the nominal
Gaussian. On a new paired2048-context truth/generated panel, numerically invert
F_c in float64 and evaluate its full12x12 Jacobian by autodiff. Compute
`log q_neural(y|c) = log Normal(z) - log abs(det dF_c/dz)` on the recovered branch.
Compare this with nominal `log q_Gaussian`, and with independent displaced-start
inverse solutions. Stop interpretation if inversion residuals or Jacobians fail.

This is a NUMERICAL SINGLE-BRANCH density diagnostic. Agreement of inverse starts
and well-conditioned Jacobians supports the calculation on this panel; it does
not mathematically certify global injectivity or exclude an unseen remote branch.
The nominal Gaussian ratio is NEVER called the exact neural-generator ratio.
All policy training still uses the learned frozen classifier, not this diagnostic.

The diagnostic gives paired learned-vs-reference BCE, logit errors on truth and
generated samples, nominal-vs-neural log-density differences, and tail agreement.
It does not replace raw held-out classifier ESS with analytic population ESS.

## Gates before policy training (fixed now)

Retain ALL existing `conditional.validity` gates, including ESS/N<=.01,
abs(log mean ratio)<=.50, baseline lower-order checks and real within-K signal.
Add instrument/tail validity:

- inverse maximum residual<=1e-8; displaced-start inverse disagreement<=1e-7;
  minimum Jacobian singular value>=.25 and no nonfinite density;
- learned BCE excess over numerical reference on the SAME diagnostic panel<=.10;
- RMS learned-logit error on truth<=1.0, to reject spurious low-ESS outliers.

Report normalization uncertainty and ESS on eight disjoint confirmation chunks,
not merely one low point estimate. These are diagnostics, not changed gates.
Normalization is not itself a DGPO-gradient claim: a constant logit offset
cancels under LOO. Do not use normalization alone to explain reward failure.

If a gate fails: `inconclusive_setup`, save the selected reward and diagnostics;
do not claim that the desired low-ESS ablation ran. If gates pass: run the
UNCHANGED conditional protocol's300-step DGPO and pathwise diagnostic arms,
from the same initial generator, with AdamW1e-4, K8, M4, no additive KL.
The pathwise arm is a different estimator and not compute matched.

Primary endpoint remains independent-context paired fixed-reward improvement,
gain>=.10 and lower95% CI>0. Use the existing decision rules. Scientific scope:
whether this learned low-ESS toy permits reward transfer; even a pass/failure
does NOT isolate low ESS as the sole cause in production.

## Scope and artifacts

CPU/local only. Read the existing source checkpoint via `--source`; retain its
generator weights exactly. Default seed17 is a pilot, not a replicated result.
Write settings, live JSONL, selected reward checkpoint, density diagnostics,
policy endpoint checkpoints when eligible, and report to a separate output.
`--smoke` is an execution check with tiny budgets; it never yields a scientific
decision. No W&B, NERSC, production edits or automatic architecture search.

## Instrument refinement: global invertibility sufficient bound

The original density report honestly marked global injectivity unproved. A
subsequent read-only bound on the fixed MLP establishes a stronger instrument:
the noisy-input spectral-norm product times1.1^3 bounds its Lipschitz constant.
For SiLU, the maximum derivative is .5+x*/4 where x*tanh(x/2)=2; x*<2.4,
so the1.1 bound is conservative. With v=sigma*MLP, every DDIM step is
`A*x+B*v(x)` and has an inverse contraction when `abs(B)*sigma*L < A`.
The source's minimum margin is .278614>0, making the composed sampler globally
bijective under this sufficient bound (spectral norms evaluated in float64).
Future diagnostic reports attach the bound/margin and set the certification flag
only when it passes. Numerical residual checks still apply. Earlier reports are
not silently rewritten, and arbitrary/evolved models are not assumed to pass.

## Bounded representation follow-up, declared after coordinate-stream result

The20k fresh-coordinate-Fourier arm completed with test BCE=.436938,
AUC=.872707, ESS/N=.212573 and log mean ratio=.033449. Normalization improved
but the intended concentration did NOT emerge. On the paired numerical-density
panel, truth logit RMS error=4.187; learned/reference BCE=.419792/.179154.
Neural-minus-nominal log-density RMS is only~.011. Fresh data alone is therefore
insufficient at this budget; do not silently increase that budget.

One separately named `--feature-mode joint3` follow-up changes ONLY the Fourier
representation, relative to the fresh-data arm. Append sin/cos of ALL C(12,3)=220
triples, ALL four sign patterns modulo overall sign, at the first harmonic.
Retain the original individual harmonics1..4 and context. No truth-selected
groups, learned truth phase, oracle ratio or mixture formula is exposed.
This is an explicit generic order-three inductive bias, not an architecture-free
test or a claim that Fourier features discover arbitrary interactions unaided.
It increases the input layer parameter count; record this confound rather than
claiming a parameter-matched representation comparison.

Keep20k fresh-data updates, source, seeds, data distribution, optimizer, nonlinear
head, standardization procedure, validation selection, confirmation panel
and ALL gates unchanged. Reusing a confirmation panel across declared variants
makes the comparison exploratory, not a pristine confirmatory replication.
Save in a separate output; no additional dictionary/architecture search in this
round. Only if all gates pass may the original300-step policy arms run.
