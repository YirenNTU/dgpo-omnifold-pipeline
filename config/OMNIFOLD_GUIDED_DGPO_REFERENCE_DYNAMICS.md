# Saturated H4 paired-refit reference-dynamics experiment

## Single question

Does a fresh saturated H4 reward and its paired current-policy reference every
ten committed DGPO updates prevent the coefficient-1 reference gradient from
canceling H4 and yield more stable cold-H4 closure than `h4rep01`?

## Why the reward and reference move together

The installed H4 log ratio is fitted with the policy at the beginning of its
round as the generated denominator.  The trainer currently uses that same
policy snapshot as the velocity-MSE reference.  Recentered reference weights
with an unchanged classifier would therefore break the checked density-ratio
pairing.  This experiment refreshes both at the same policy and preserves that
pairing.  It does not change the H4 logit, calibrated LOO advantage, detached
gate, K=8 expectation, reference formula, reference coefficient, or AdamW.

## Matched control and intervention

The control is the independent-seed fixed-reward run `h4rep01`.  Both arms use
the same `c4a91e07` step-1110 weights-only source and seed2 bundle.  The control
keeps its bootstrap reward/reference through step 50.  This arm installs a
fresh, independently initialized, saturated 2-fold x 2-repeat H4 stack and the
exact current policy as its paired reference after steps 10, 20, 30, and 40.
AdamW moments and scheduler state are retained across those installations.

## Measurements and decision

- Cold selection-blind H4 audits: steps 0/10/20/30/40/50.
- Every audit must be saturated and use at least 1,000 optimizer updates.
- Exact gradient-transfer traces: steps 1/2/5/10/20/30/40/50.
- Matched pre-refit and post-install gradient-conflict probes show whether each
  paired refresh returns the reference gradient to zero without changing the
  H4 direction.
- Primary within-run endpoint:

```text
mean(gap30, gap40, gap50) - gap0 < 0
```

where `gap = abs(cold_H4_AUC - 0.5)`.  Total variation and worst positive
excursion across the six audits describe stability against the completed
`h4rep01` control.  They do not replace the predeclared closure endpoint.

## Launch

From an allocated 16-GPU Ray cluster:

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter python3 scripts/train_dgpo_h4_saturated_refit10.py
```

The run logs live to W&B ID `h4ref10` and writes a final
`reference_dynamics_endpoint.json` beside its checkpoints.
