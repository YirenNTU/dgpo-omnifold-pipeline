# Frozen diffusion Cij sampling stress test

## Wide targeted moment scan (2026-10-02)

```bash
shifter python3 -u scripts/run_tau_cij_sampling_bias.py config/tau_cij_sampling_bias_wide.yaml
```

Reuses the saved 16-GPU inference of `vnk1rs3v` (raw DGPO step890 and original
pretrain epoch214) without generating or training again. Creates a new result
directory and W&B run; does not overwrite the original report. No Ray or GPU
allocation is required for this numerical analysis.

For each of nine full-truth C moments, solve a stable exponential tilt
`P(event) proportional to original_MC_weight * exp(lambda * event_C_contribution)`
to reach targets -0.5,-0.3,-0.1,0,0.1,0.3,0.5 in expectation. Then actually
resample N complete paired events with replacement, 100 times. All drawn
events have UNIT analysis weight; never apply MC or tilt weights again.
Nominal also samples proportional to original MC weight with unit analysis
weight. This weighting convention differs from the original bin-multiplier
scan, but represents the same nominal weighted population in expectation.

This is a targeted angular-moment stress test, not a certified physical
spin-state injection. Selection, charge/analyzer conventions and decay-channel
mixtures can affect interpretation. All other C moments and six B moments are
recorded, not held fixed. B is defined explicitly as +3<a_i/kappa_a> and
+3<b_i/kappa_b>, using signed parquet kappas; no acceptance correction or extra
charge sign is imposed. Do not call these certified physical polarizations.

Read `bias/direct/C/full` for actual truth versus generated C (ideal y=x),
`bias/direct/signed_bias/full` for generated-minus-truth (ideal zero), and
`bias/direct/B/full` for associated B changes. `matched` references isolate
the fixed-energy reconstruction approximation. The nominal summary plots show
actual values, not delta C. Intervals are resampling ranges, not errors on the
100-repeat mean. More repeats stabilize the Monte Carlo summary, not the
underlying independent-event count or fixed diffusion-draw uncertainty.
No fit/calibration is applied. All models use identical indices.

`bias/direct/C_matrix_values/full` displays three side-by-side nominal Cij
matrices (truth, pretrain, DGPO), with signed numeric entries and a shared color
scale. These are actual full-panel values, not errors, delta C or repeat means.
The `matched` counterpart uses the matched truth reference. For the resampled
target scan, continue to use `bias/direct/C/full` with actual truth on the x axis.

`bias/target_support` lists requested/expected/achieved C, sampling ESS, maximum
probability, top1% probability mass and expected unique-event fraction.
Unsupported targets are explicitly skipped, never clipped or retargeted.
ESS/N below 0.1 is flagged in plots/table (a diagnostic warning, not a universal
physics threshold). Repeated events do not create new independent information.

Primary question: does each model follow the actual full-truth C across this
predeclared range? Judge signed bias and response, not only a smaller nominal
matrix error. The validation panel is development data, not a blinded final
coverage measurement. No fresh checkpoint is substituted in this matched replay.

## Original bin-based scan

Run on an existing 16-GPU Ray allocation:

```bash
shifter python3 -u scripts/run_tau_cij_sampling_bias.py config/tau_cij_sampling_bias.yaml
```

The launcher resolves and snapshots the DGPO root's `checkpoints/last.ckpt`
at launch, and the original 10% supervised diffusion checkpoint. It prints
their resolved paths, epochs and steps. Each uses its own saved runtime and
raw `state_dict` (never EMA), including checkpoint normalization. This is an
overall pipeline comparison, not an isolated architecture ablation.

Inference uses the existing complete filtered validation panel (119002 events),
16 GPUs, 1024 events/GPU/batch, label 0, DDIM20 and one unselected prediction
per condition. Truth is never a model input. Both models use the same event
order and inference seed. Predictions are saved and reused for all tests.

For each of the nine truth cosine products, divide [-1,1] into six equal-width
bins. Resample events with replacement with bin multipliers
0.5,0.7,0.9,1.1,1.3,1.5 or the reverse. Together with nominal this gives 19
variants, each repeated 100 times. Use exactly the same sampled indices for
truth and both models. Retain original MC weights in Cij, but do NOT multiply
the injection factors again. This is a shape stress test, not a complete
physical BSM model.

Report all nine matrix components and Frobenius errors:

- Absolute: norm(C_generated - C_truth).
- Response: norm((C_generated,shift - C_generated,nominal) -
  (C_truth,shift - C_truth,nominal)).
- DGPO minus pretrain for each error: negative means improvement.

The full truth reference uses stored truth four-vectors with the repository's
common k/r/n convention. A separate matched truth reference uses the same
fixed-energy reconstruction as generated deltas, to expose representation
limitations. Bins always use full truth products. No classifier ratio is used.

Every launch gets a fresh output directory and W&B run. The report and response
plots appear in W&B; checkpoints and sample arrays stay on scratch. To repeat
analysis without inference:

```bash
shifter python3 -u scripts/run_tau_cij_sampling_bias.py config/tau_cij_sampling_bias.yaml \
  --analysis-only /pscratch/sd/y/yiren/Ztautau/tau_cij_sampling_bias/probe-REPLACE
```

Intervals describe resampling variability of this fixed development panel and
fixed predictions, not training uncertainty or independently certified coverage.
A truth-dependent selection changes the conditional target. Even an ideal
nominal conditional generator need not follow arbitrary changes with frozen
weights: failure here is sensitivity to distribution shift, not by itself proof
of faulty training. Do not calibrate the models on these injected answers.
