# H4 AdamW direction-transfer ablation (`h4opt01`)

## Question

At the completed `h4xfer01` step-20 anchor, does the restored AdamW state turn
an exact useful H4-plus-reference gradient into a direction that is worse for
the same frozen installed four-member H4 ensemble?

## Fixed scientific contract

- Restore `h4xfer01` step 20, reward round 1, including policy, round reference,
  four-member H4 reward, AdamW moments, scheduler clock, and
  parameter groups.
- Recompute one 8,192-event, K=8, eight-timestep production gradient with the
  installed reward, leave-one-out advantage, and coefficient-1 velocity-MSE
  reference term.
- Preserve production gradient clipping. Do not fit a classifier, commit an
  optimizer step, modify a checkpoint, transform the reward, or change the
  objective.
- Compare native AdamW, native AdamW without weight decay, zero first moment
  with the saved second moment and clock, fresh AdamW, and the raw gradient.
- Scale every direction to the same symmetric VP-path distance as native AdamW
  at parameter RMS `1e-6`.
- Score plus, zero, and minus with the frozen installed step-20 H4 ensemble on
  2,048 identities and eight common-random-number rollout seeds. Gradient, VP,
  and judge identities are disjoint. `h4xfer01` independently measured an
  installed-to-fresh H4 gradient cosine of 0.896 at step 20. The cold monitor
  was intentionally transient because `warm_start_classifier: false`, so its
  weights are not present in the checkpoint and cannot be replayed.

## Decision

The primary paired statistic is the native-plus judge gap minus the raw-plus
judge gap over the eight seeds. A positive 95% paired interval establishes
that optimizer transformation, rather than the mathematical objective alone,
loses useful H4 direction. The no-decay, zero-first-moment, and fresh-state
arms then locate the responsible AdamW component. If raw and native are not
separated, the next experiment should return to reward drift or nonlinear
multi-step accumulation.
