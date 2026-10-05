# Is conditional scale error worth targeting?

Read-only diagnostic, not a new ratio estimator for deployment. No classifier
training, diffusion updates, candidate generation, cap tuning or physics input.
Uses completed fzekzrmr K64 panel, scored by frozen pzq0nl1i FiLM classifier:
119,002 filtered validation conditions, raw step1110 generator, 16-GPU source.
Does not use the still-running fresh-negative classifier m0nfa76x.

## Run on NERSC

From the existing ml_pipeline checkout:

```bash
shifter python3 -u scripts/diagnose_tau_conditional_normalization.py \
  config/conditional_tau_normalization_diagnostic.yaml
```

CPU replay uses 16 CPU threads; no GPU inference or Ray allocation is needed.
The user launches it. A new W&B run and unique output directory are created.
Original samples, scores and checkpoints are untouched.

## Fixed comparisons

At K32 and K64 prefixes, compare unweighted, raw, cap30, same-pool normalized,
and disjoint-half normalized scores. With r=exp(logit), same-pool uses r/Z_full.
Disjoint normalization uses r_A/Z_B and r_B/Z_A, where Z is the arithmetic mean
ratio on the other half. All arms retain base event weights and are globally
normalized. Splitting uses saved draw order, not scores or truth.

Also report A_using_Z_B versus raw_A, and B_using_Z_A versus raw_B separately.
These directional comparisons preserve each evaluation half's within-condition
relative scores. The combined arm can additionally change the relative mixture
of halves; its improvement alone is not sufficient evidence of scale correction.

Same-pool normalization enforces event masses by construction. Its mass TV is
therefore NOT evidence of correct ratios. Symmetric disjoint event mass is
proportional to base_weight*cosh(log Z_A-log Z_B): noisy denominators can worsen
mass closure. Inverse-estimated Z is biased, and disjoint candidates do not make
the combined arms independent. K32/64 denominator sizes are 16/32, respectively.

## Endpoints and interpretation

- Replay original K64 unweighted/raw/cap30 Cij, Frobenius error and event ESS
  before marking the new result complete. Keep inherited signed kappa convention.
- Report all nine Cij components, paired whole-event bootstrap error changes
  against unweighted/raw/cap30, and approximate simultaneous component bands
  for the 36 normalization-minus-raw contrasts (2 arms x 2 K x 9 components).
- Report half-denominator log correlation, RMS disagreement, quantiles,
  within-condition ESS, event/category mass TV and numerical zero weights.
- Save full report, per-event denominators and one comparison figure. W&B gets
  summary metrics, all component contrasts, report and figure; large event arrays
  stay on disk. No truth-based arm is selected or deployed.

Improvement only with same-pool normalization is ambiguous: a finite-K
self-normalization effect may explain it. Consistent improvements with disjoint
normalization and stable half denominators would support condition-scale error
as a useful target for this frozen head, not prove it is the dominant bottleneck.
No improvement does not prove relative score error: support and finite-K noise
remain alternatives. Component tradeoffs must be inspected even if the matrix
norm improves. Cij closure alone is not full conditional tau closure.

Bootstrap holds fitted models and candidate panel fixed; it excludes model-fit
uncertainty and unseen tails. Previously inspected conditions are not a new
independent-event test. The diagnostic neither recertifies physics conventions
nor claims removal of a condition-only offset fixes within-condition ordering.
