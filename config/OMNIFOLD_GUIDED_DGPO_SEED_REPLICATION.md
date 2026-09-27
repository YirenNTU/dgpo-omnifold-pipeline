# Independent-seed classifier-closure replication (`h4rep01`)

## Single question

Does the predeclared `h4xfer01` + `h4xfer02` 50-update classifier-closure
result reproduce under an independent stochastic seed bundle?

## Matched control and intervention

The matched control is the exact full trajectory formed by `h4xfer01` steps
0–20 and its full-state continuation `h4xfer02` steps 21–50. The replication
starts weights-only from the same `c4a91e07` policy step-1110 checkpoint and
retains the intentional full-state restart boundary at step 20.

Only these explicit seeds change:

| Source | Control | Replication |
| --- | ---: | ---: |
| reward/ensemble fit | `20260920` | `20261020` |
| fixed reward-fit population | `42` | `314159` |
| cold audit panel, rollout, and classifier | `20260921` | `20261021` |
| gradient-conflict panel | `20260915` | `20261015` |
| TARP null trials | `20260820` | `20261022` |

The current DGPO trainer does not expose its process-level policy-rollout RNG
as a YAML seed. Fresh Ray workers therefore provide an independent rollout
stream, while the five controllable stochastic sources above are explicitly
pinned for replay.

## Fixed training contract

- Same c4a91e07 step-1110 weights-only source and epsilon-sweep-selected LR.
- One four-member H4 reward (two repeats by two cross-fit folds) installed at
  bootstrap and held fixed through step 50.
- Same K=8 calibrated leave-one-out DGPO reward objective.
- Same coefficient-1 `velocity_mse` soft reference and native AdamW.
- No refit, hard trust boundary, projection, optimizer reset, reward
  transformation, early stopping, or checkpoint selection.
- Cold H4 audits at steps 0/10/20/30/40/50, each requiring at least 1,000
  optimizer updates and saturation.
- Exact gradient-transfer telemetry at steps 1/2/5/10/20/30/40/50.
- Sixteen Ray workers with one GPU each and live W&B run `h4rep01`.

## Two-phase execution

Phase 1 runs steps 0–20 and writes a complete recovery checkpoint. Phase 2
loads that checkpoint in `resume` mode and preserves policy, EMA, AdamW
moments, scheduler, installed H4 stack, round reference, adaptive state, and
all clocks while running steps 21–50. Both phases append to W&B `h4rep01`.

## Predeclared result table and decision

| Policy step | Cold H4 AUC | Gap `abs(AUC-0.5)` | Audit updates | Saturated |
| ---: | ---: | ---: | ---: | ---: |
| 0 | — | — | — | — |
| 10 | — | — | — | — |
| 20 | — | — | — | — |
| 30 | — | — | — | — |
| 40 | — | — | — | — |
| 50 | — | — | — | — |

The single primary statistic is computed within this replication:

```text
mean(gap30, gap40, gap50) - gap0
```

The replication passes only when this value is negative and all four audits
used in it are cold, saturated, and trained for at least 1,000 optimizer
updates. Step-50 improvement and monotonicity are reported as secondary
descriptions. If it passes, the unchanged H4-guided target has an independent
replication. If it fails, inspect seed-sensitive critic/reference dynamics
before changing the optimizer or objective.

## Launch

From an active 16-GPU Ray allocation:

```bash
shifter python3 scripts/train_dgpo_h4_seed_replication.py
```

The driver completes phase 1 before starting phase 2. A preempted phase can be
resumed with `--phase 20` or `--phase 50` after confirming which checkpoint
directory contains `last.ckpt`.
