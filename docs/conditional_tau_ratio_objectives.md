# Conditional tau ratio-objective ablation

Question: with the exact completed BCE `443eg16h` representation and samples,
does a direct ratio objective improve full-population reweighting, and does
non-negative risk correction add value? **BCE is reused, never refitted.**

Two fresh conditional MLP heads: MLC and nnUKL. Same raw step1110 frozen
features, six relative angles, same prepared file (read-only symlink), same
train/validation/test identities and base weights. No new generation,
backbone fit, EMA, cap, tempering, MMD penalty, ESS penalty or DGPO updates.
The existing train/validation filter manifests are checked before fitting.

Settings inherit the completed BCE manifest and are checked against its YAML:
16 GPUs, 1024 paired conditions/GPU, seed 42, maximum 250 epochs, AdamW 2e-4 -> 1e-5
cosine, weight decay .001, dropout .05, hidden 128. Minimum validation BCE
selects the checkpoint in all arms; no external Cij/test selection. Matching
the selector intentionally compares training objectives, not optimally tuned
estimation procedures. No multi-seed sweep. Test inference is also 16 GPU
shards at 1024 conditions/GPU; identity merging rejects missing/duplicate rows.

User-requested early stopping applies to BOTH new arms: validation BCE,
patience 25 epochs, min_delta 1e-4, minimum 1000 optimizer steps. Significant
improvements reset patience; non-improving epochs only count after the minimum
step gate. Absolute lowest validation BCE still saves the checkpoint, even
when the improvement is smaller than min_delta. Cosine keeps its original
250-epoch horizon and is not compressed when training stops. The reused BCE
ran the full budget, so this is no longer an equal-compute comparison with
BCE. Manifest/W&B explicitly record this difference. MLC vs nnUKL shares the
same stopping policy, but realized fit steps can differ. Log both total fit
steps and selected-checkpoint steps.

## Mathematics

Let P = E_truth[exp(s)], Q = E_generated[exp(s)], T = E_truth[s]. Network output
s is log p/q in both arms, and inference uses the raw generated logit.

- MLC/UKL: `(Q - T)/2` (additive constants omitted).
- nnUKL: `(-T + c*P + max(0, Q-c*P))/2`.
- Equivalent: MLC + `max(0, c*P-Q)/2`.

The common 1/2 matches the existing paired BCE scale. Event weights are
normalized once over the fit population, never by random minibatch totals.
Global DDP batch expectations are formed BEFORE the nonlinear maximum; doing
it per GPU and then averaging is a different objective. A local autograd
surrogate accounts for DDP gradient averaging. Float64 is used only for
exponential risk arithmetic; the network remains float32. Nonfinite risks or
gradients abort without silently changing the objective. No gradient clipping
is added. Training-only zero-coefficient MMD diagnostics are skipped in both
new arms; validation geometry diagnostics stay unchanged.

`nnukl_c: 0.01` is a fixed exploratory choice, **not a certified ratio bound**.
Population preservation requires true `sup(p/q) < 1/c = 100`, plus the
paper's other conditions. We do not know this bound. Do not tune c on test
Cij. If correction never activates, this is effectively MLC and says nothing
about the usefulness of an active correction. This is not a hard ratio cap.
The implementation uses the corrected objective (plain descent), not the
paper's optional gradient-ascent heuristic. It can still fail from overlap,
misspecification or optimization; it is not guaranteed to raise ESS or fix Cij.

References: [MLC study](https://arxiv.org/abs/2305.10500),
[nnBD / UKL instance, Sec. 3](https://proceedings.mlr.press/v139/kato21a/kato21a.pdf).

## Endpoints and decision

Primary: full-population matched Cij Frobenius error versus unweighted AND
saved BCE, with paired event bootstrap of their differences. The nnUKL arm
also compares with a completed matched MLC from this family when available.
Keep the adopted signed per-event analyzing-power estimator unchanged. This
does not independently certify its physical convention or detector acceptance.

Supportive: conditional tau moments by channel/visible-pT strata, raw ratio
ESS, max mass, top-1% mass and mean ratio. Report all nine Cij entries.
Training W&B: BCE/AUC, objective, mean ratios in both classes, signed risk
component, correction magnitude, activation fraction, correction/MLC logit
gradient norm. These distinguish inactive correction from an unsuccessful
active intervention. Objective values across BCE/MLC are not comparable.

Lower Cij error with a negative paired 95% interval supports improvement at
this fixed sample/model level, not full conditional closure. Reject an ESS-only
"success". Check tau-stratum regressions before further use; don't select the
best-looking test metric. The external test has been inspected in previous
rounds: outcomes are exploratory. Bootstrap excludes classifier-training
uncertainty and can be fragile at very low ESS. A fresh weighted audit remains
a separate next step if endpoints improve; the output is compatible with
`scripts/audit_conditional_tau_ratio.py` via its `source_directory` YAML field.

## Commands on the existing NERSC Ray allocation

```bash
shifter python3 -u scripts/run_tau_ratio_objectives.py \
  config/conditional_tau_ratio_objectives_10pct.yaml prepare

shifter python3 -u scripts/run_tau_ratio_objectives.py \
  config/conditional_tau_ratio_objectives_10pct.yaml all
```

Or replace `all` with `mlc` / `nnukl` to launch individually. `all` is strictly
MLC then nnUKL, not BCE. Each invocation creates a separate output and a new
W&B run, never overwrites the BCE checkpoint. The shared output root records
`mlc_latest_completed.json` / `nnukl_latest_completed.json`; these are only
updated after all endpoints succeed. If MLC fails (including numerical
divergence), `all` still attempts nnUKL: stabilizing that failure is part of the
question. The overall command exits nonzero after both attempts if either
arm failed, and never presents a failed arm as a completed comparison.

Local validation cannot read `/pscratch`: `prepare` verifies those actual
source artifacts on NERSC without Ray, W&B or training. The user submits all
jobs; this change does not request allocations or start remote training.

## Observed MLC result — 2026-09-30

Run [s01j1pez](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/s01j1pez)
finished successfully but failed the ratio-quality endpoints. Read the full
200-epoch history and saved `cij_comparison.json`, not just the final summary.
Early stopping fired at epoch 200 / 4400 updates; minimum-validation-BCE
selection restored epoch 175 / 3850 updates (BCE 0.412639).

On the identical 119002 external events, selected MLC versus saved BCE:

| Metric | BCE 443eg16h | MLC s01j1pez |
| --- | ---: | ---: |
| Test BCE | 0.390485 | 0.414503 |
| Test AUC | 0.903227 | 0.893355 |
| ESS (events) | 903.30 | 2.79 |
| Largest normalized weight | 3.05% | 48.99% |
| Top 1% weight mass | 30.75% | 99.46% |
| Cij Frobenius error | 1.11056 | 28.15947 |

Unweighted Cij error remains 0.52131. Selected MLC generated mean ratio is
126.69, despite training-batch generated means remaining approximately one.
The final-epoch validation metrics are different from restored-checkpoint test
metrics and must not be mixed: at epoch 200 validation max mass is 96.81%.

The failure develops while validation BCE improves: epochs 50/100/175 have
validation BCE 0.42950/0.41843/0.41264 but validation ESS fractions
5.99%/0.002415%/0.003582%, respectively. At epoch 175 the largest validation
weight already carries 47.87% of mass. Thus this is not merely failing to
restore the best BCE checkpoint. BCE-only stopping/selection does not control
exponential tail generalization in this experiment. The pattern supports
empirical direct-ratio overfitting/extrapolation; it does not independently
prove physical support holes or an implementation error.

The paired bootstrap difference versus BCE is +27.0489 [12.8710, 62.8359],
but with ESS below three its coverage is not trustworthy enough for precise
significance claims. The gross tail concentration and endpoint regression
are the decisive evidence. Budgets differ (BCE 250 epochs; MLC 200).
The companion nnUKL run `sfisiaz2` was still running when inspected; do not
conclude that risk correction succeeds or fails from this MLC result.

## Completed nnUKL result — 2026-09-30

Run [sfisiaz2](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/sfisiaz2)
finished as `Does risk correction help? | frozen tau head | nnUKL | raw 1110`.
Same frozen raw step-1110 cache, samples, splits, head and optimizer schedule;
`c=0.01`. It stopped at epoch 206 / 4532 updates and selected the absolute
minimum-BCE checkpoint at epoch 200 / 4400 updates (validation BCE 0.409939).
Patience last reset at epoch 181: the later absolute improvement was smaller
than min_delta=1e-4, so selection at 200 and stopping at 206 are consistent.

Read all 206 epoch records and the saved Cij/moment reports. Correction was
active for one global minibatch each at epochs 40, 107, 148 and 179: four of
4532 updates. The final summary's zero activation is not a whole-run count.
At each activation epoch validation ESS increased, with a BCE excursion;
for example epoch 147 -> 148 changes ESS fraction 0.928% -> 11.735% and BCE
0.41270 -> 0.44649. Afterwards concentration grows again: selected epoch 200
validation ESS is 0.703%, versus 3.624% at epoch 179.

On the same 119002 external events, selected nnUKL has test BCE 0.411294,
AUC 0.894128, ESS 886.91 (0.7453%), largest mass 2.0107%, top-1% mass 40.029%,
and generated mean ratio 1.19281. MLC's corresponding ESS 2.79 and mean ratio
126.69 are dramatically worse, but nnUKL does not improve the BCE control's
ESS 903.30 or top-1% mass 30.747%. Tau direction-moment error is 0.063752
(unweighted 0.001831; BCE 0.049194).

Cij Frobenius error is 2.023006, versus unweighted 0.521308, BCE 1.110560
and MLC 28.159467. Paired bootstrap difference nnUKL-minus-unweighted is
+1.501699 [0.904711, 3.763849]; nnUKL-minus-BCE is +0.912446
[-0.342947, 2.802321], so worsening relative to BCE is not statistically
resolved by this bootstrap. Previously inspected test, fixed models/candidates,
no refit uncertainty, and extreme MLC ESS limit inference.

Hypothesis update: an active non-negative correction can materially suppress
the catastrophic MLC tail in this trajectory, even with very rare activations.
It is not sufficient for conditional tau/Cij closure at this c and selector.
This is not an inactive-correction negative test, and does not establish
generator support failure. No need to rerun merely to test whether correction
ever activates. Ratio-quality selection and residual conditional bias remain
unresolved; do not deploy this ratio as a demonstrated physics improvement.
