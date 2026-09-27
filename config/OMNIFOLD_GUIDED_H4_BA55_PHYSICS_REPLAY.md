# H4 BA55 physics-first replay

## One question

Does the previously validated H4 BA55 policy direction improve the target
physics distributions at normalized policy RMS `3e-5`?

This is an artifact replay of the completed `aqbszk1r` diagnostic. It uses the
saved H4 BA55 two-fold, one-repeat reward, fixed candidate logits, and saturated
independent H4 judge. It performs zero classifier fits and zero committed policy
updates.

## Primary endpoint

For eight common-random-number rollout seeds, compare the plus policy step with
the unchanged policy using the mean JSD of:

- `tau_a_delta_theta`
- `tau_a_delta_phi`
- `tau_b_delta_theta`
- `tau_b_delta_phi`

Support requires a negative mean paired JSD change, a negative 90% bootstrap
upper bound, plus beating zero in at least 6/8 seeds, plus beating minus in at
least 6/8 seeds, and no nonfinite generated candidates. Individual target JSDs,
response summaries, and the independent H4 judge gap are recorded as secondary
diagnostics. Saturated classifier AUC is not a gate.

## Run

```bash
shifter python3 scripts/diagnose_ba55_physics_replay.py \
  config/dgpo_10pct_c4a91e07_h4_ba55_physics_replay_v2.yaml
```

The corrected W&B display name is `Does BA55 H4 improve target physics? | RMS 3e-5 | corrected replay`
and its stable run ID is `h4b55phys2`. The dedicated entry point fails before
launch if the underlying replay script does not contain the physics endpoint.
