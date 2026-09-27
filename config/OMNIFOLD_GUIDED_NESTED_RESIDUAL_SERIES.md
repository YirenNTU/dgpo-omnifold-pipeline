# Conditional H2 critic series

## Question

Can a capacity-controlled conditional H2 critic recover held-out residual
structure and provide an actionable DGPO direction, without the unrestricted
capacity and overfitting seen in the earlier H2/H4 classifiers?

All arms use the fixed `c4a91e07` policy checkpoint at step 1110, the same
identity partitions, two folds, one repeat, candidate panels, rollout seeds,
RMS grid, and independently saturated H4 judge.

| Arm | Reward function | W&B ID | Display name |
|---|---|---|---|
| old | standalone legacy control | `oldbase02` | `Is legacy signal exhausted? \| old classifier \| loss-selected \| fixed policy` |
| rank2 | direct rank-2 conditional H2 critic | `h2r2direct` | `Can rank-2 recover residual signal? \| direct H2 critic \| fixed policy` |
| rank4 | direct rank-4 conditional H2 critic | `h2r4direct` | `Does rank-4 add useful signal? \| direct H2 critic \| fixed policy` |

The earlier `oldr2h201` and `oldr4h201` pilots are excluded. Both stopped in
the old-base stage before emitting any residual-fit point. In `oldr2h201`,
training balanced accuracy reached `0.769`, validation balanced accuracy stayed
at `0.5085`, and validation BCE reached `1.241` at step 1625. In `oldr4h201`,
the corresponding old-base values at step 1200 were `0.734`, `0.5120`, and
`1.021`. These runs establish that the old feature class is exhausted at this
policy endpoint; they do not test either conditional residual.

`h2band01` did test H2 features and overfit, but it retained the full nonlinear
Fourier classifier. At step 1100 its train/validation balanced accuracies were
`0.668/0.5078` and validation BCE was `1.026`. The direct low-rank critic below
is a capacity ablation of that failure, rather than a repeat of the same model.

## Model and optimization contract

For rank `r`, the direct score is

```text
s_r(x,z) = <A_r c(x), B_r phi_H2(x,z)> / sqrt(r)
```

`c(x)` concatenates frozen pretrained diffusion features: the global event
token and the masked mean of visible PET tokens. `phi_H2(x,z)` contains the
exact H2 periodic tau-pair features. The candidate factor is zero initialized,
so training starts from the balanced null classifier.

There is one fit stage. The legacy decoder and output are frozen at zero, and
only `A_r` and `B_r` are trained with balanced truth/generated BCE. This removes
the overfitting old-base stage while preserving the density-ratio objective.
No reward rank transform, z-score, clipping, or alternative DGPO loss is used.

Each direct critic may run up to 3,000 optimizer steps and cannot stop before
1,000. `restore_best` selects the minimum held-out BCE checkpoint, so a later
memorizing state cannot replace a better calibrated density-ratio estimate.

W&B gives one independent step axis per live fold-1 curve:

```text
classifier_fit/reward_rank2_residual/*
classifier_fit/reward_rank4_residual/*
```

The old control uses `classifier_fit/reward_old_base/*`. Fold 2 is retained in
the report and checkpoints without being mixed into the live plot.

## Run

Run rank 2 first. Rank 4 is needed only if rank 2 generalizes but lacks enough
held-out discrimination or H4-judged actionability.

```bash
shifter python3 scripts/diagnose_reward_interface.py \
  config/dgpo_10pct_c4a91e07_nested_residual_rank2.yaml
```

Optional matched controls:

```bash
shifter python3 scripts/diagnose_reward_interface.py \
  config/dgpo_10pct_c4a91e07_nested_residual_old.yaml

shifter python3 scripts/diagnose_reward_interface.py \
  config/dgpo_10pct_c4a91e07_nested_residual_rank4.yaml
```

Each run logs training curves, held-out classification, candidate ordering,
gradient alignment, and the multi-seed H4-judged actionability sweep in real
time. It performs no policy update or reward installation.

## Decision rule

Rank 2 is eligible for a closed-loop DGPO pilot only when:

1. held-out BCE improves below the null value and held-out AUC separates from
   chance without a growing train/validation gap;
2. at least one RMS has a negative 90% upper confidence bound for
   `plus_gap - zero_gap` and satisfies the predeclared seed-consistency gates;
3. candidate and policy-gradient alignment with the independent H4 judge are
   finite and positive.

Run rank 4 only if rank 2 passes generalization but its actionable correction
is too weak. If rank 2 already passes, the lower-rank model is preferred.

After all three reports exist, run:

```bash
shifter python3 scripts/compare_nested_residual_series.py \
  --old /pscratch/sd/y/yiren/Ztautau/c4a91e07_old_classifier_matched_actionability_v2/trajectory_report.json \
  --rank2 /pscratch/sd/y/yiren/Ztautau/c4a91e07_direct_rank2_h2_actionability_v1/trajectory_report.json \
  --rank4 /pscratch/sd/y/yiren/Ztautau/c4a91e07_direct_rank4_h2_actionability_v1/trajectory_report.json \
  --output /pscratch/sd/y/yiren/Ztautau/c4a91e07_conditional_h2_comparison_v1.json
```
