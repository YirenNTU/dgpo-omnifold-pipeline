# Five-fold OOF residual ablation

This ablation compares the production one-repeat two-fold residual stack with
one identity-stable five-fold split. It is not five two-fold repeats and not
five full-data in-sample fits.

Each residual iteration trains five classifiers. Every event is scored only by
the fold model that did not train on that identity. That single held-out logit
is the training increment. On the outer validation population and DGPO
candidates, all five fold models are unseen and receive coefficient `0.2`.

Warm starts match the same iteration and fold. A two-fold checkpoint is not a
valid five-fold warm start.

## Fixed-policy gate

Config: `config/dgpo_10pct_oof_ensemble_diagnostic.yaml`

The two arms share the exact generated step-320 pool, outer 80/20 split,
classifier architecture, optimizer, tempering, closure rule, and training
seeds. The focused screen skips the residual-weight intervention controls.

```bash
shifter python3 scripts/diagnose_residual_weights.py \
  config/dgpo_10pct_oof_ensemble_diagnostic.yaml --check-only

shifter python3 scripts/diagnose_residual_weights.py \
  config/dgpo_10pct_oof_ensemble_diagnostic.yaml
```

Advance the five-fold arm only if both training seeds show consistent
outer-validation weighted-physics improvement, no larger fit-versus-OOF gap,
and no material ESS or weight-tail regression. AUC or lower member variance
alone is insufficient.

## DGPO paired pilot

- Control: `config/dgpo_omnifold_ztautau_10pct_oof_r1f2_control.yaml`
- Ensemble: `config/dgpo_omnifold_ztautau_10pct_oof_r1f5_ensemble.yaml`

Both start weights-only from the same step-320 live policy with fresh clocks and
isolated checkpoint, Ray, and W&B paths.

Both use soft velocity-MSE trust with coefficient 1, disable the adaptive hard
boundary, and apply a 20-update policy-LR warmup after every accepted reward
installation. This keeps the trust regularizer while avoiding hard-boundary
step collapse and the full-LR refit transition seen in the legacy 10% history.

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_10pct_oof_r1f2_control.yaml \
  -- --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_visible_rest_l2_forward_refit_p4_8_16_24_fullfit_gradlife_step320_seed42_oof_r1f2_control/ray_results

shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_10pct_oof_r1f5_ensemble.yaml \
  -- --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_visible_rest_l2_forward_refit_p4_8_16_24_fullfit_gradlife_step320_seed42_oof_r1f5_ensemble/ray_results
```

Compare through step 100 before extending to step 300:
`val_ztautau/jsd/current/*`, `staleness/raw_*`, residual closure/ESS,
fold disagreement, gradient conflict, trust distance, and classifier GPU-hours.
The five-fold arm performs 2.5 times as many classifier fits per residual
iteration, so runtime is part of the result.
