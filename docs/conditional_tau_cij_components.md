# Why do individual Cij components move differently?

Prepared diagnostic, not yet run on NERSC. Primary question: on the fixed
`fzekzrmr` fresh-K64 panel, do the raw/cap30 changes arise from concentrated
event contributions, broadly distributed contributions, or shared events that
help some Cij elements while harming others? No new cap is selected.

## Evidence motivating the diagnostic

Completed `fzekzrmr` (2026-09-30) used the same119002 filtered validation
conditions,64 fresh candidates each, raw step1110 generator and frozen
`pzq0nl1i` FiLM classifier; no classifier fits/policy updates. Cij norm errors:
unweighted0.5654395, raw0.8175171, cap30 0.3658000. Capped-minus-unweighted
paired95% interval[-0.3248844,-0.0475342]. Cap30 event ESS63185.5 and
max candidate mass5.1281e-6; removing its leading-influence candidate changes
error0.3658000 to0.3659126. Seven component absolute-error point estimates
improve; rk/rn worsen. This is a sampling replication, not new-condition
generalization, and does not prove that every high ratio is misestimated.

## Frozen inputs and command

Source: `/pscratch/sd/y/yiren/Ztautau/conditional_tau_cap_confirmation/confirm-fedfec09a0`.
The launcher verifies completion, W&B run identity, raw/frozen16-GPU K64
provenance, unchanged cap30, completed filtered119002-event manifest, and exact
input/candidate ID order. Recomputed truth and all three Cij/ESS endpoints must
reproduce the saved source report. Normalization and stored Cij features are
used unchanged; no raw-data fallback or reconstruction convention change.

```bash
shifter python3 -u scripts/diagnose_tau_cij_components.py \
  config/conditional_tau_cij_components.yaml
```

This is **CPU arithmetic over saved16-GPU results**, not single-GPU inference.
No Ray/GPU/model loading, generation, classifier fits or training submission.
Uses16 CPU threads; typically requires a few GB RAM for the full saved panel.
Creates a new output directory and online W&B run; sources remain untouched.
Optional `--no-wandb` runs only local reporting.

## Endpoints and uncertainty

For each raw/cap30 arm and all nine axes kk,kr,kn,rk,rr,rn,nk,nr,nn:

`absolute_error_change = abs(C_arm-C_truth) - abs(C_uniform-C_truth)`.

Negative is better.2000 paired event bootstraps keep the truth and all64
candidates of each condition together. Report pointwise percentile intervals
and an approximate simultaneous95% band over **all18 arm/component contrasts**:
the95th percentile of the maximum absolute centered bootstrap deviation gives
one common half-width (all quantities use identical units). Label improvement
only when the simultaneous upper bound <0, worsening only when lower bound >0;
otherwise unresolved. This is not an exact finite-sample guarantee: absolute
errors are nonsmooth at zero, tails can be unseen, models are fixed, and this
population was already inspected. Bands do not cover post-hoc event/category
searches or supply causal attribution.

## Exact event accounting

Let b_i be normalized base mass, w_ik globally normalized arm candidate mass,
a_i=sum_k w_ik, m_i the unweighted mean of candidates within event i, and U
the global unweighted Cij vector. Define:

```
within_i  = sum_k w_ik * (g_ik - m_i)
between_i = (a_i - b_i) * (m_i - U)
delta_i   = within_i + between_i
sum_i delta_i = C_arm - U
```

This separates redistribution within a condition from changing total weights
between conditions; it does NOT normalize the deployed weights per event.
The centering makes accounting invariant to an additive observable offset.
For each component, r_A=C_arm-truth, r_U=U-truth, multiply every delta_i by
`(r_A+r_U)/(abs(r_A)+abs(r_U))`, defining0 when the denominator is0.
Their sum equals the **absolute-error change exactly**, including overshoot.
The same scaling separates within/between contributions. This is descriptive
secant accounting, NOT independent effects from deleting particular events.

Report positive (harmful) and negative (helpful) sums, event fractions, share
of harm from the top1/10/100/1% events, and top50 events on both sides for each
component. A reported remainder after subtracting selected terms is explicitly
NOT a renormalized deletion estimate. No events are removed. Category groups
show contributions and original/weighted mass; they do not veto spin results.

Tradeoff matrix entry(j,l) is the fraction of helpful contribution to j from
events with positive harmful contribution to l. Missing helpful mass gives
null, not evidence of no conflict. This summarizes observed event overlap,
not mutually incompatible physical targets or inability of a better ratio to
improve both. Shared parameters/normalization and noisy empirical targets
prevent causal interpretation.

## Outputs and decisions

- `component_intervals.png`: nine contrasts for raw and cap30 with family bands.
- `event_accounting.png`: within/between split and tradeoff heatmaps.
- W&B tables: all component intervals, concentration, event IDs, category sums.
- `component_report.json`: full source/settings, arithmetic, intervals, limitations.
- `event_contributions.npz`: all event-level arrays, kept on scratch (not uploaded).

Few events accounting for most harm motivates a tail-control training test.
Broad harmful contributions motivate conditional-ratio/representation checks,
but do not alone prove population bias. Shared helpful/harmful events identify
where a scalar cap may trade off elements, not a demonstrated impossibility.
No thresholds automatically decide which classifier or regularizer to deploy.
