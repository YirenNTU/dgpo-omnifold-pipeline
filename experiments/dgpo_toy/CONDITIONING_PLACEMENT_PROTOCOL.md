# Pretrained diffusion: does early Fourier injection interfere with learning?

Status: prepared, scientific training not launched. This is a local MLP toy,
not an EveNet/PET implementation change or proof about attention.

## One question

Does injection position alone change held-out velocity MSE when adding the
same Fourier adapter to a working pretrained backbone?

Earlier frequency experiments started from random weights; they cannot test
interference with an existing representation. Here only injection position
differs between Early and Late. No frequency, LR, normalization, data, or
adapter-capacity advantage is bundled into Late.

## Data and model

Use the existing moving Gaussian with default `f(c)=sqrt(2)*sin(4*pi*c)`,
`c~Uniform[-1,1]`, conditional width .35, and fixed32768/8192/16384 train,
validation/test events. The SAME samples and SAME function are used before
and after the fork. No new distribution or pretraining-only easy target.

The velocity model has input embedding `[x_t, sqrt(3)*c, time_features] ->
Linear -> GELU`, two residual MLP blocks with pre-LayerNorm, and a shared
`LayerNorm -> Linear -> SiLU -> Linear` velocity head, multiplied by sigma(t).
Width64, FP32. It learns ordinary sampled velocity targets, not the oracle.

The adapter in both Fourier arms is identical:

`8 Fourier features -> Linear -> SiLU -> LayerNorm -> Linear(no bias)`.

Fixed frequencies1..4, paired sin/cos with unit population variance. Only the
last projection is zero-initialized; internal layers use ordinary initialization.
Fourier reads observed c only. No target or clean sample enters this branch.

## Source readiness, not a convergence claim

First pretrain the no-Fourier model on the SAME problem. Save the first
best-validation checkpoint that satisfies all prespecified conditions:

- Validation sample velocity MSE is at least50% below initialization.
- Validation oracle-excess velocity MSE lies in [.01,.05].

This intentionally selects a useful but imperfect source; it is not a claim
that the baseline is fully converged. The analytic oracle is used ONLY to
ensure a suitable starting point and for diagnostics, never as network input
or training supervision. Test data never select the source.

If patience20 with min_delta1e-4 (or an optional epoch cap) is reached without
a valid source, stop with `inconclusive_source_not_ready`. Do not silently
change thresholds, function, or checkpoint until the desired conclusion appears.

## Three matched branches

| Arm | Placement |
| --- | --- |
| none | Continue the source without Fourier; adapter inactive/frozen. |
| early | Add adapter residual after input GELU, before the two backbone blocks. |
| late | Add identical adapter residual after the backbone, before the head. |

Copy the SAME source and adapter internal weights. Verify exact velocity
equality at the fork. Early/Late have identical active parameter counts.
All backbones and heads remain trainable. Late leaves the backbone's forward
input unchanged, but does NOT block its backward gradients or prevent drift.

All three reset AdamW together: lr1e-3, weight decay.001, clipping1; no special
adapter LR, schedule, EMA, classifier, DGPO, KL or generation diagnostics.
Every optimizer update uses identical sampled minibatch/time/noise across arms.

Train all arms to the same available update budget: joint patience20 after
each has stopped making1e-4 improvements. Select each arm's exact minimum
validation MSE, including fork-step0. Default no total-step/epoch cap.
Optional `--max-epochs` caps EACH stage and never implies convergence.

## Measurements and prespecified interpretation

Primary contrast: Late minus Early independent test velocity MSE, conditional
on the same source. Continued None is required to interpret harm/preservation.
Report all three paired contrasts with approximate pointwise95% intervals
and material margin .005, inherited from previous toys.

Evidence supporting early-placement interference in this toy requires:

1. Early-minus-None lower interval > .005 (material harm).
2. Late-minus-None upper interval < .005 (noninferiority within margin,
   NOT an insignificant-difference claim).
3. Late-minus-Early upper interval < -.005 (material rescue).

Otherwise report the observed contrasts without forcing an interference
diagnosis. Both Fourier arms improving fails to reproduce harmful injection.
Both worsening does not isolate early placement. Single seed17 is a pilot;
three seeds still share the same generated dataset. Pointwise intervals are
not simultaneous uncertainty or training-seed uncertainty.

Secondary diagnostics, not alternative endpoints: fixed train-probe MSE,
oracle excess, time/condition MSE slices, clip norm/fraction, adapter residual
RMS divided by injection-token RMS, and backbone-output RMS change relative
to frozen source. Feature drift alone is not damage; connect it to MSE.

This MLP has no particle attention, masks, or variable-length tokens. A pass
motivates testing placement in EveNet, not claiming its bottleneck is solved.
Frozen-backbone and adapter-LR experiments are explicitly deferred.

## Launch locally (user starts training)

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
/opt/miniconda3/envs/MyEve/bin/python -u \
  -m experiments.dgpo_toy.conditioning_placement \
  --output artifacts/dgpo_toy/conditioning_placement_pilot1 \
  --seeds 17
```

Outputs: fixed dataset, `progress.jsonl`, `report.json`, `source.pt` plus
best/last raw weights for each branch. No automatic resume or remote upload.
Tests run tiny bounded optimizer loops only; these are not scientific results.
