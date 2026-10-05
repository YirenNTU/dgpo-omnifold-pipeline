# Does geometry change the wrong part of the weights?

This is a diagnostic of completed results, **not another classifier training run**.
The user launches it on NERSC:

```bash
shifter python3 -u scripts/diagnose_tau_weight_factor_swap.py \
  config/conditional_tau_weight_factor_swap.yaml
```

It creates a **new W&B run** and a uniquely named output directory. It does not
modify the source results. All inference was already performed with 16 GPUs;
this step uses 16 CPU threads for saved-array arithmetic and needs no Ray workers
or new GPU inference. Add `--no-wandb` only for offline local verification.

## Frozen inputs and question

- A: context256, trained bound30, `zrv2yfgt`.
- B: geometry inputs, same architecture width and bound30, `n3jjqcn7`.
- Same `fzekzrmr` raw-step1110 K64 panel, 119002 filtered validation conditions.
- Completed train/validation filter manifests, identities/order, model protocol,
  inherited panel endpoints and AA/BB Cij/ESS endpoints are checked.
- No retraining, generation, ratio cap selection, physics convention changes,
  or tuning to this test panel.

For each model, globally normalized candidate masses are factored as
`w[i,k] = m[i] * pi[i,k]`, where `m[i] = sum_k w[i,k]` and
`pi[i,k] = softmax(log_ratio[i,:])[k]`. Event base weights are included in `m`.
Zero-base-weight events have zero mass; their proportions are computed without
dividing by zero. All four combinations sum to one globally:

| Arm | Event mass | Within-event candidate proportions |
|---|---|---|
| AA | A | A |
| AB | A | B |
| BA | B | A |
| BB | B | B |

The hybrids are **not calibrated conditional density ratios** and need not obey
the original pointwise bound30 constraint. They are not deployment candidates.
Applying another cap would break the factor-swap experiment, so none is applied.

## Predeclared endpoints and interpretation

Primary: Cij Frobenius error changes **AB−AA** and **BA−AA**, with a joint
approximate 95% bootstrap max-deviation band over these two contrasts.
Negative means closer to truth. All nine component absolute errors are retained;
their 6×9 contrast family gets a separate simultaneous band.

- AB improves: B's within-event proportions help **at A's event masses**.
- BA worsens: B's event masses hurt **at A's within-event proportions**.
- The combination above supports an event-mass tradeoff explaining why B's
  improved BCE did not translate to better global Cij, on this panel.
- AB worsens: the problem is not solely B's event masses; its candidate
  allocation can itself worsen spin closure.
- Effects depend on the other factor: inspect `BB−BA`, `BB−AB` and the
  interaction `BB−BA−AB+AA`; do not declare one universal culprit.
- Intervals cross zero: **unresolved**, not evidence that the factors are equal.

Secondary outputs: each arm vs unweighted, signed Cij contrasts, candidate/event
ESS, group-mass TV, and category × fit-defined visible-pT group Cij errors.
Groups are descriptive, not separately significance-tested. Group residuals
sum to the global signed residual; grouped error norms do not equal global error.
An interaction in the error norm can arise from the nonlinear norm itself;
the report also supplies signed matrix contrasts, and neither establishes a
physical causal interaction.

2000 paired whole-event bootstrap draws keep truth and all64 candidates together.
This estimates event-sampling uncertainty for fixed classifiers/candidate panels;
it does not estimate training uncertainty or remove finite-K bias, and does not
correct for earlier inspection of this test set. Non-smooth absolute errors make
the simultaneous bands approximate. Previous Cij conventions are inherited, not
independently recertified. The test can localize an observed weighting tradeoff,
**not uniquely prove whether its origin is missing information, ratio calibration,
generator support, or classifier optimization**. It also does not identify why
cap30 helped the earlier unbounded model: both A and B are already bounded.

W&B receives two figures, arm summaries, contrasts, all component intervals,
group table and the full JSON report. Event-level numerators/masses and bootstrap
matrices stay in the NERSC output NPZ for follow-up; no large source sample upload.
