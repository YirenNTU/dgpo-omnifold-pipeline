# H4 gradient-transfer trace (`h4xfer01`, continuation `h4xfer02`)

## Question

On the exact `c4a91e07` step-1110 lineage, does the always-on coefficient-1
`velocity_mse` reference gradient cancel a still-useful H4 gradient during the
native AdamW trajectory?

This is a 20-update diagnostic. It does not change the applied DGPO objective,
reward transformation, ESS tempering, AdamW state, ensemble size, or rollout
protocol.

## Frozen trajectory

- Source: `c4a91e07`, policy step 1110, `weights_only`.
- Reward: one fresh H4 2-repeat x 2-fold ensemble, held fixed for all 20 updates.
- Applied loss: `L_H4 + 1.0 * L_reference`.
- Optimizer: the existing native AdamW path.
- No hard trust boundary, projection, refit, rollback, or extra regularizer.
- W&B: `h4xfer01`, online with `resume=allow`.

## Measurements

At applied update endpoints 1, 2, 5, 10, and 20, the training graph records:

- `g_H4`, the exact installed-H4 loss gradient;
- `g_ref`, the exact unweighted soft-reference gradient;
- the reconstructed `g_H4 + lambda * g_ref` and the actual unclipped gradient;
- `theta_before - theta_after`, the native AdamW descent displacement including
  momentum, preconditioning, per-group learning rates, and weight decay;
- read-only projection ratios for `lambda in {0, 0.1, 0.2, 0.5, 1}` and the
  critical coefficient where the H4 projection crosses zero.

Cold H4 audits run at policy steps 0, 10, and 20. Each must execute at least
1,000 classifier optimizer updates and reach its validation plateau. Step-10
and step-20 audit weights are used once for a read-only installed-H4 versus
fresh-H4 gradient-alignment probe, then discarded. They never become reward.

## Required result table

| Policy step | cold H4 AUC gap | audit fit updates | `cos(g_H4,g_ref)` | total-on-H4 projection | critical lambda | AdamW-on-H4 cosine | installed-vs-fresh H4 cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | | | | | | | |
| 1 | — | — | | | | | — |
| 2 | — | — | | | | | — |
| 5 | — | — | | | | | — |
| 10 | | | | | | | |
| 20 | | | | | | | |

The step-0 row has a cold audit but no applied AdamW transition. The trace row
label 1 describes the actual transition from policy step 0 to step 1.

## Decision rule

- If installed H4 stays aligned with fresh H4 while total-on-H4 crosses below
  zero, deterministic soft-reference cancellation is confirmed.
- If the total gradient remains H4-aligned while the native AdamW displacement
  loses that alignment, optimizer state or preconditioning is the bottleneck.
- If installed H4 loses alignment with the fresh H4 judge before cancellation,
  critic staleness is the leading mechanism.
- If none occur while cold H4 AUC does not improve, the next diagnostic must
  test finite-step policy realization rather than another classifier change.

Near-0.5 AUC with fewer than 1,000 audit updates is invalid and the launcher
fails if an audit does not saturate.

## Same-trajectory persistence continuation

`h4xfer02` full-state resumes the completed `h4xfer01` step-20 checkpoint and
continues the same trajectory through step 50. It preserves policy, EMA,
native AdamW moments, the installed four-member H4 reward, its paired round
reference, all adaptive clocks, and every stochastic seed. Bootstrap and
resume refit are disabled, so reward round 1 remains installed throughout.

Cold saturated H4 audits and exact gradient-transfer measurements run at
steps 30, 40, and 50. The predeclared persistence endpoint is
`mean(gap30, gap40, gap50) < gap0`, where the `h4xfer01` step-0 gap is
`0.36228927164638947`. This tolerates individual oscillations while requiring
the late trajectory to remain better on average. Step-50 gap and monotonicity
are reported separately.

```bash
shifter python3 scripts/train_dgpo_h4_gradient_transfer_trace.py \
  --config config/dgpo_omnifold_ztautau_10pct_h4_gradient_transfer_trace_resume20_to50.yaml
```
