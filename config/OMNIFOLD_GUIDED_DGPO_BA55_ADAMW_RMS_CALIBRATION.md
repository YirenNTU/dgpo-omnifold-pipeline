# BA55 native-AdamW RMS-calibration experiment

## Question

Does applying the complete native AdamW proposal at the BA55 replay's selected
full-parameter RMS of `3e-5` improve the one-update cold-H4 endpoint relative
to the ordinary native AdamW step?

This tests step-size transfer through AdamW. It does not change the DGPO loss,
reward transformation, candidates, reference term, AdamW moments, or relative
per-parameter preconditioning.

## Paired arms

Both arms:

- start weights-only from `c4a91e07` step 1110;
- cold-fit one repeat by two cross-fit folds of the nonlinear H4 classifier;
- select the first checkpoint whose oriented held-out balanced-accuracy 95%
  lower bound reaches `0.55`;
- use raw LOO reward, fixed tempering `0.75`, K=8, and fresh AdamW;
- make exactly one DGPO optimizer proposal;
- run cold H4 audits at step 0 and step 1, requiring at least 1,000 classifier
  optimizer updates and saturation;
- stream classifier curves, optimizer displacement, and audit metrics to W&B.

The native arm applies AdamW normally. The treatment first advances the same
AdamW moments, then multiplies the complete proposed displacement by one
scalar so that

```text
RMS(theta_applied - theta_old) = 3e-5.
```

Because AdamW's moment update does not depend on LR and its parameter update is
linear in a common LR multiplier, this produces the same direction and state
as rerunning that proposal with every parameter-group LR multiplied by the
logged calibration scale.

## Primary endpoint

Compare

```text
step1_gap - step0_gap
```

where `gap = abs(cold held-out H4 AUC - 0.5)`. Both audits must be saturated
and contain at least 1,000 fit updates. The calibrated arm passes if its paired
gap change is lower than the native control and its logged applied RMS is
within the implementation tolerance of `3e-5`.

The frozen independent judge from `h4b55lr1` remains a mechanistic diagnostic;
it does not replace this fresh best-response endpoint.

## W&B

```text
ba55nat1  Native AdamW one-step control | BA55 two-fold H4 | cold H4 audit
ba55rms1  RMS-calibrated AdamW at 3e-5 | BA55 two-fold H4 | cold H4 audit
ba55lr10  AdamW LR 1e-5 one-step | BA55 two-fold H4 | cold H4 audit
```

Both runs use group `H4 policy projection` and independent W&B clocks for
classifier-fit curves.

The `ba55lr10` arm is an additional matched learning-rate ablation. It leaves
RMS calibration disabled and changes only the native policy LR from `1e-6` to
`1e-5`. Its realized update RMS determines whether the tenfold LR moves the
policy far enough while retaining the production AdamW update.

## Commands

Run the two arms in separate 16-GPU allocations or windows:

```bash
shifter python3 scripts/train_dgpo_ba55_rms_calibration.py \
  --config config/dgpo_omnifold_ztautau_10pct_h4_ba55_m2_adamw_native_1step.yaml
```

```bash
shifter python3 scripts/train_dgpo_ba55_rms_calibration.py \
  --config config/dgpo_omnifold_ztautau_10pct_h4_ba55_m2_adamw_rms3e5_1step.yaml
```

Add `--check-only` to either command for fail-closed preflight.

Run the matched tenfold-LR arm with:

```bash
shifter python3 scripts/train_dgpo_ba55_rms_calibration.py \
  --config config/dgpo_omnifold_ztautau_10pct_h4_ba55_m2_adamw_lr1e5_1step.yaml
```
