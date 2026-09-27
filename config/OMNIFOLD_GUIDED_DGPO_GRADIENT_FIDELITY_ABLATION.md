# OmniFold-guided production DGPO gradient-reproducibility audit

Status: completed on 2026-09-15. W&B run: `h4grad01`.

Implementation:

- runner: `scripts/diagnose_production_gradient_reproducibility.py`;
- config: `config/dgpo_10pct_c4a91e07_h4_production_gradient_reproducibility.yaml`;
- contract tests: `scripts/test_diagnose_production_gradient_reproducibility.py`;
- W&B run ID: `h4grad01`;
- local output: `/pscratch/sd/y/yiren/Ztautau/c4a91e07_h4_production_gradient_reproducibility_v1`.

## Single research question

> Is one current production-sized DGPO gradient already reproducible across
> independent Monte Carlo realizations?

This is the necessary check before adding more sampling. It preserves the
existing mathematical DGPO objective exactly and performs no optimizer step.

## What production already averages

The current trainer does not use one candidate or one scalar sample per update.
One production optimizer step already contains:

- `K=8` candidates per event for the LOO advantage;
- `num_train_timesteps=8` diffusion-time/noise substeps whose gradients are
  accumulated;
- event batches distributed over 16 DDP workers and averaged across ranks;
- event microbatch accumulation when needed;
- one AdamW update only after all event chunks and timestep substeps finish;
- shared diffusion noise across the K candidates of an event at a given
  timestep, reducing candidate-comparison noise.

The earlier `t8darve0` split-half cosine near `0.487` used only 1,024 gradient
events and four timesteps. It motivates a production-matched audit but does not
prove the normal training gradient is still noisy.

## Objective contract

For every realization, call the unchanged production loss:

```text
A[i,e] = r[i,e] - mean_{j != i} r[j,e]
Delta[i,e] = stopgrad(L_cur[i,e] - L_ref[i,e])
w[e] = stopgrad(sigmoid(beta/K * sum_i A[i,e] * Delta[i,e]))
loss = mean_{e,i}(w[e] * A[i,e] * L_cur[i,e])
```

Keep the reward, calibrated-LOO scale, K, beta, masks, reference, event gate,
time distribution, candidate generator, and all regularizers identical to the
resolved production configuration. Do not introduce z-score, rank, clipping,
softmax tilt, a new baseline, or a different advantage.

Each realization must compute all eight timestep substeps and each substep's
nonlinear gate exactly as `train_step` does. A realization is one complete
pre-AdamW accumulated production gradient.

## Common anchor

- Source W&B run: `c4a91e07`.
- Load mode: `weights_only`.
- Source policy step: `1110`.
- Checkpoint:
  `/pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/checkpoints/dgpo-epoch=110-next_ep=111-step=1110.ckpt`.
- Frozen reward: completed calibrated-LOO four-member H4 reward-interface
  ensemble.
- Workers: 16 GPUs, matching production DDP reduction.
- K: 8.
- Training timesteps accumulated per realization: 8.
- Preserve the source coefficient-1 soft `velocity_mse` reference-trust term.
  Its first-order gradient should be zero at the exact current=reference anchor;
  log it explicitly as a contract check.
- No classifier fit, reward refit, policy optimizer step, scheduler step,
  staleness action, rollback, or hard trust.
- Validate checkpoint path, policy step, tensor compatibility, runtime schema,
  and W&B provenance. Do not add a SHA contract.

## Four independent production gradients

From the unchanged anchor, compute:

```text
g1, g2, g3, g4
```

Each gradient uses an independent production-compatible draw of events,
candidates, timesteps, and diffusion noise. Restore the anchor and clear all
gradients before every realization. Do not advance AdamW.

Each realization uses 512 events on each of 16 ranks, or 8,192 events in its
distributed mean. The four batches are disjoint (32,768 gradient events total).
The signed-probe panel uses another 2,048 events drawn from the remaining
`final_audit` partition with an independent selection seed.

Define the high-precision diagnostic mean:

```text
g_bar = (g1 + g2 + g3 + g4) / 4
```

`g_bar` is used only to study convergence of the same objective. It is not a
new reward or loss.

## Primary endpoint

The primary statistic is the mean of the six pairwise cosine similarities:

```text
mean_pairwise_cosine(g1, g2, g3, g4)
```

Also report the minimum pairwise cosine and the cosine between the two
independent half means:

```text
cos((g1 + g2)/2, (g3 + g4)/2)
```

Predeclared interpretation:

| Gradient reproducibility | Conclusion |
| --- | --- |
| mean >= 0.90 and minimum >= 0.80 | Production Monte Carlo variance is not the main failure. Do not increase sampling. |
| mean 0.70–0.90 | Residual variance is material but may not explain the small policy effect. Use the signed probes below. |
| mean < 0.70 | Production gradient estimation is unstable enough to be a plausible bottleneck. |

These thresholds apply to the full trainable parameter vector. Always report
per-block results because a stable large block can hide an unstable
reward-relevant adapter or generation head.

## Same-objective signed probes

Create normalized `+/- 1e-6` parameter-RMS probes for `g1`, `g2`, `g3`, `g4`,
and `g_bar`. Evaluate them with the existing fixed independent H4 judge on a
common event/noise panel.

This secondary check asks whether gradient disagreement changes the observable
direction:

- each plus probe should beat its paired minus probe;
- the four individual plus probes should agree in sign;
- `g_bar` should not be worse than the median individual gradient if variance
  reduction is useful.

Do not train cold H4 audits in this first audit. A fresh H4 classifier is only
warranted if `g_bar` is substantially more stable and gives a larger
fixed-judge improvement than the individual production gradients. This avoids
nine expensive classifier fits before production variance is established.

## Variance decomposition diagnostics

Using the same four realizations, report:

- global and per-block gradient norms and pairwise cosines;
- cosine of each `g_i` to `g_bar`;
- rank-local cosine across the eight diffusion-time draws on a small,
  logging-only diagnostic subset;
- distribution of the detached event gate `w_e` across realizations;
- requested parameter RMS and applied flat-gradient scale of every signed
  probe;
- fixed-judge AUC gap for every plus/minus probe;
- generated-sample non-finite fraction.

Per-event gradient energy, reward-member gradient cosines, and VP-path distance
are deferred. They would add separate estimators to this first one-question
audit and are only warranted if the four full gradients disagree.

## W&B contract

```text
project: ytchou97-university-of-washington/nu2flow-RL
id: h4grad01
name: c4a91e07_h4_production_gradient_reproducibility_v1
group: c4a91e07_h4_projection_diagnostics
```

Required metric prefixes:

```text
gradient_repro/objective_contract/*
gradient_repro/g1/*
gradient_repro/g2/*
gradient_repro/g3/*
gradient_repro/g4/*
gradient_repro/gbar/*
gradient_repro/pairwise/*
gradient_repro/probe/*
gradient_repro/primary/*
```

Required summary keys:

```text
gradient_repro/primary/mean_pairwise_cosine
gradient_repro/primary/min_pairwise_cosine
gradient_repro/primary/half_mean_cosine
gradient_repro/primary/gbar_fixed_judge_gap_delta
gradient_repro/primary/production_variance_status
gradient_repro/diagnosis
```

Stream the completion and metrics of every production gradient and every signed
probe to W&B. Upload the resolved config manifest and final report as one
artifact. Save the four gradients and `g_bar` locally as float16 replay vectors;
do not upload the several-hundred-MB vector file by default.

## Execution

Inside the initialized 16-GPU Ray allocation:

```bash
shifter python3 scripts/diagnose_production_gradient_reproducibility.py \
  config/dgpo_10pct_c4a91e07_h4_production_gradient_reproducibility.yaml
```

Path/schema validation without starting Ray or W&B:

```bash
shifter python3 scripts/diagnose_production_gradient_reproducibility.py \
  config/dgpo_10pct_c4a91e07_h4_production_gradient_reproducibility.yaml \
  --check-only
```

## Decision table

| Result | Diagnosis | Next experiment |
| --- | --- | --- |
| Gradients highly reproducible | Current sampling is already sufficient; the expected DGPO direction is intrinsically weak in parameter space. | Test target-preserving preconditioning or functional/natural-gradient geometry. |
| Gradients disagree and `g_bar` clearly improves the fixed judge | Monte Carlo variance materially loses H4 signal. | Run normal DGPO with more accumulation, then use saturated cold H4 audits. |
| Gradients disagree but `g_bar` does not improve the fixed judge | Averaging alone cannot recover a useful direction. | Decompose classifier-member versus event/candidate variance while keeping the target fixed. |
| Individual and averaged plus probes lose to minus probes | Loss/probe sign or implementation contract is inconsistent. | Stop and audit the exact backward path. |

## Observed result

- Mean pairwise cosine: `0.95524`.
- Minimum pairwise cosine: `0.91110`.
- Cosine between independent two-gradient means: `0.97506`.
- Every individual plus probe improved the fixed judge; every minus probe
  worsened it.
- `g_bar` changed the fixed-judge gap by `-0.00517`, close to the individual
  gradients and far smaller than the `h4cfw001` ideal fixed-support change.
- Candidate non-finite fraction: zero.

The result selects the first decision-table branch. The production gradient is
already reproducible; increasing same-objective Monte Carlo accumulation alone
is not justified. The next causal target is optimization geometry or local
policy conditioning.

## Scope

This audit answers whether the **existing production gradient estimator** is
reproducible. It does not test long-run convergence and does not change the
DGPO optimization target. Its result decides whether additional sampling is
scientifically justified.
