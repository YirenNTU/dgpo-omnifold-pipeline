# Conditional reward retention, 2026-09-29

Toy-only, bounded to three reviewed rounds; no production edits, remote jobs,
uploads, changed rewards, new seeds, LR sweep, or modified DGPO objective.
The user authorizes local diagnostic iterations. Existing truth, learned frozen
full-truth H4, velocity half-MSE coefficient 1 and raw weights remain fixed.

## Question

Do updates lose earlier reward gains in the same conditions, or induce losses
elsewhere? Does a controlled reduction of cross-condition parameter sharing
help, IF measured interference is present? This is a hypothesis, not an assumed
explanation. Fourier benefit does not by itself prove the remaining limitation.

## Round 1 — existing endpoints, no fitting or updates

Use completed conditioning_closed_loop_20260926 round04 raw/Fourier and its
matched one-factor controls, endpoints 0/25/100/300/1000. All use the same
frozen critic, 64 midpoint conditions and 512 paired draws per condition.
Primary descriptive comparison is 25 -> 1000, not a selected best checkpoint.
Report early gain, later change, net gain, and exact accounting of gross later
loss into erased positive early gains versus additional damage; later gains
complete the accounting identity. This clipped accounting is noisy/descriptive,
not an unbiased causal attribution.

Avoid winner-selection regression to the mean: assign early-beneficiary groups
using half the candidate draws, then estimate later changes using the other
half; swap halves and average event contributions. Keep this distinct from
the exact full-sample accounting. Conditions are a deterministic grid: any
condition-bootstrap interval describes grid heterogeneity, not a random test
population or training-seed uncertainty. No distribution-closure claim.

Strong aggregate regression is reproduced only if early reward gain lower95
exceeds .01, subsequent change upper95 is below -.01, and final gain is <50%
of early gain. A non-significant change is not failure/equivalence.
If not reproduced, do not change the target until it fails. Inspect conditional
interference as a diagnostic of the existing task and label it separately.

The same analysis accepts native checkpoint-transfer NPZ files, aligning every
event ID and rank across endpoints. Production inputs are currently unavailable
locally; SSH authentication failed. No production event-level conclusions until
those files are supplied. The existing scalar report is not a replacement.

## Review gate for subsequent rounds

Write a result/decision before each next round. Only choose a targeted local
probe supported by round1. A gradient interference test must distinguish
surrogate parameter-gradient dot products from actual held-out generated reward.
Use fixed condition partitions selected without outcomes, independent training
and evaluation noise, unchanged learned H4/reference and no truth-mode inputs.
Independent specialists are an intentionally non-deployable upper-bound
diagnostic, not proof that a larger architecture would generalize or a proposed
production replacement. Do not claim rescue without a valid reproduced failure.

Stop after three rounds or an earlier decisive/inconclusive boundary. Browse
primary research at the end and state what it supports versus what is only a
candidate intervention. Fresh adequately fit classifiers remain necessary for
distribution claims; do not launch them solely to answer fixed-reward retention.

## Completed bounded campaign

All three rounds completed on2026-09-29. Result and decision ledger:
`artifacts/dgpo_toy/reward_retention_20260929/REPORT.md` and each round's
`DECISION.md`. No strong aggregate toy regression was reproduced. Independent
per-bin reward derivative estimates were not stable enough to establish
interference. Keep the production-cause hypothesis unresolved; do not extend
this campaign by forcing a new truth/difficulty. Primary literature checked
after the rounds in `REFERENCES.md`. Production paired arrays await download.
