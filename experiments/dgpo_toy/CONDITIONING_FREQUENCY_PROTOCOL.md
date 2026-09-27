# Can frequency-bank mismatch create a velocity-MSE plateau?

Status: implemented and awaiting user launch. No new scientific fit submitted.

## Question and prior evidence

The completed `conditioning_mse_pilot1` used seed17 and a true sine at k=4,
already present in its k=1..4 Fourier bank. Test MSE improved from .365568
to .355450; epoch10 validation MSE was1.014096 versus .357823. Linear and
bump tasks did not reproduce material Fourier harm. Those results support
task-dependent learning speed, not a general solution for real diffusion.

This round asks whether replacing the conditioning-frequency bank, at fixed
feature count and LR, can rescue slow supervised velocity learning on a more
demanding conditional function. It does NOT test a learning-rate restart,
prove the cause of the real run's plateau, or test high-dimensional copulas.

## Intuitive data

Retain the one-dimensional moving Gaussian:

`c ~ Uniform[-1,1]; y | c ~ Normal(f(c), .35^2)`.

Only the rule for how the cloud center moves changes:

- `periodic_high`: `f(c)=sqrt(2)*sin(12*pi*c)`. A fast wave; its true k=12
  occurs in the high-frequency bank. This is a deliberately favorable control.
- `chirp`: center/rescale `sin(pi*(8*c+3*c^2))` to mean0 and variance1.
  Local phase slope in k units is `8+6*c`, increasing from2 to14 as the knob
  turns. It is not a single fixed-frequency sinusoid. Normalization uses a
  deterministic65,536-point midpoint quadrature, never training/val/test data.

Same Gaussian width and target variance; same oracle conditional velocity
formula and irreducible risk as the previous toy. No change to the training
loss or truth access. Model inputs remain only `(x_t,t,c)` and transforms of c.

## Four matched arms, two functions

| Arm | Fixed k values | Extra features |
| --- | --- | --- |
| raw | none | inert adapter for stored-size matching |
| fourier (low) | 1,2,3,4 | 8 |
| fourier_mid | 2,4,6,8 | 8 |
| fourier_high | 3,6,9,12 | 8 |

Use `sqrt(2)*sin(k*pi*c)` and `sqrt(2)*cos(k*pi*c)`, all unit population
variance. Raw c remains available in EVERY arm. The shared MLP is identical;
the additive projection is zero-initialized. Check exact initial velocity
equality across all four arms. All three Fourier arms have equal active
parameter counts; raw's inert adapter does not match active feature capacity.

Raising the bank scale changes BOTH its range and spacing: these are scaled
sparse banks, not nested dense spectra. A benefit supports bank alignment,
not the stronger claim that larger maximum k always helps. Low frequencies
can also compose through a nonlinear MLP; a missing explicit frequency is
not proof that the architecture cannot represent it.

Use the previous matched data sizes, FP32, hidden64, AdamW lr1e-3, weight
decay.001, clip1. No LR/scheduler/normalization change, classifier, reward, KL,
EMA, sampling or physics metrics. Draw each batch/time/noise only once for
all four arms. Data are fixed within case; all seeds share that dataset.

## Stop, measurements and decisions

Every arm receives the same updates. Stop the group when ALL arms have gone
20epochs without an absolute1e-4 improvement over their respective patience
anchors; no default step/epoch cap. Exact best validation MSE selects the
checkpoint, including epoch0. Test MSE is evaluated only after selection.

Record fixed train-probe/validation velocity MSE every epoch, oracle excess
MSE and time/condition slices, clip fraction, and the first patience-trigger
epoch per arm. That first plateau is a historical patience trigger, NOT a
proof of permanent saturation; training may later improve. Optional budget
caps are explicitly marked as not convergence.

Primary prespecified contrast per function: `fourier_high minus fourier`
independent test velocity MSE. Retain the prior paired pointwise95% interval
and absolute .005 material margin. Mid-minus-low and comparisons to raw are
secondary; do not select the best bank on test and call it the primary result.

Evidence consistent with frequency-bank rescue requires:

1. Low-bank validation exhibits a patience plateau whose best-validation
   checkpoint so far still has oracle excess >.01, not merely irreducible
   noise or one bad late iterate.
2. High-minus-low test MSE upper interval is below -.005.
3. High-bank selected-checkpoint train-probe MSE is also lower.

If high improves MSE but low never exhibits that plateau, report a frequency
benefit WITHOUT claiming the requested failure mode was reproduced. If high
only improves train MSE, flag a generalization tradeoff. If none helps, bank
expansion is not sufficient at the tested settings; this does not prove the
model is incapable of learning. These are finite-protocol observations.

Single seed17 is a pilot. Three seeds17/23/41 can assess initialization/noise
stream robustness, not new-dataset robustness. Intervals are conditional on
trained models, not simultaneous across all contrasts. No unique EveNet
bottleneck claim is authorized by this toy.

## Local launch (user starts training)

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
/opt/miniconda3/envs/MyEve/bin/python -u \
  -m experiments.dgpo_toy.conditioning_mse \
  --frequency-ablation \
  --output artifacts/dgpo_toy/conditioning_frequency_pilot1 \
  --seeds 17
```

This runs eight models, jointly paired by function. To replicate later, use a
new output and `--seeds 17 23 41`. Do not launch the replication automatically.
For a design-only preview, add `--preview-only` (zero optimizer updates).

Outputs: `design.png`, `mse_curves.png`, `progress.jsonl`, `report.json`, fixed
datasets, best/last raw weights with explicit frequency-bank metadata.
`pairs[case/seed].comparisons.fourier_high_minus_fourier` is the primary
contrast; the retained pair-level legacy `comparison` describes low versus
raw. Top-level `primary_contrast` and each case's `decisions.contrast` explicitly
select high-minus-low for this suite. `rescue_checks_by_seed` records the three
predeclared failure/rescue checks without treating a pilot as replication.

The original three-function experiment defaults remain unchanged. No existing
results are overwritten; all toy code/output stays excluded from NERSC uploads.
