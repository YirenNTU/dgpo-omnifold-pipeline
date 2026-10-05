# One residual classifier after context256 reweighting

Status: implemented and locally tested; the user launches on NERSC. No remote
job submitted. This is a new experiment, not an executed positive result.

## One question

Can a fresh classifier learn a **generalizable correction to the remaining
joint (visible condition, tau-pair) discrepancy**, after the saved context256
bounded classifier has already reweighted the generated samples?

Recent evidence motivates, but does not establish, this mechanism:

- `zrv2yfgt`: context256/bound30 is the retained comparator. On the fixed K64
  external panel its Cij error was 0.25648, off-diagonal error 0.21112.
- Explicit products `gp8z7sjn` did not establish an improvement over this
  comparator. More input features are not bundled into this experiment.
- `fri1gd2b`: nine-coefficient moment calibration nearly closed calibration
  moments, but selection cross-term error changed 0.24621 -> 0.27605 for
  strength 0.1. Its baseline fallback is not evidence of a useful correction.
- The old fresh weighted audit only measured a residual; it did not multiply
  the newly learned ratio back into the weights.

This new test uses weighted BCE, **not a Cij loss or moment calibration**.
It does not refit the first-stage classifier or update the generator/DGPO.

## Ratio algebra, including the easily missed constant

Write z=(x,tau pair), and let p be truth and q the raw1110 generator joint
distribution, with the same base event measure. The first classifier's
bounded odds are w1(z). Define Z1=E_q[w1] and q1=w1*q/Z1.

The new head uses balanced, globally normalized weighted BCE:

    L = 1/2 E_p[softplus(-s2)] + 1/2 E_q1[softplus(s2)].

Its ideal optimum is s2=log(p/q1). Therefore:

    raw cumulative log ratio = s1 + s2 - log(Z1)
    deployed log ratio       = min(log(30), raw cumulative log ratio).

Estimate log(Z1) ONCE from the residual FIT rows and save it in `stack.json`
and `residual/best.pt`. Do not estimate it from external truth or normalize
separately within each condition. Omitting Z1 does not change self-normalized
uncapped weights, but **does change which samples hit an absolute cap30**.

Training class weights have mean one separately over each full split. There
is no random minibatch normalization and no weighting of the truth class by
w1. All truth/generated candidates for an event stay in the same split.
Fitting and the fresh audits use the existing K1 samples. K64 is an independent
fixed-candidate physics evaluation; it does not supply 64 times the negative
class prior. The head sees one (condition, candidate), never a truth/candidate
comparison pair or an event ID as input.

The second head uses the same FiLM context256, candidate64, depth3 architecture
and frozen raw1110 features. Its **residual logits are unbounded**; the cap is
on the final cumulative ratio, not on each factor. This is a constrained,
biased ratio estimate after capping; it is not guaranteed exact closure or a
solution to absent generator support. Changing the condition marginal under
w1 is allowed and monitored: s2 is a JOINT residual ratio, not automatically a
pure conditional ratio.

## Frozen sources and splits

- First-stage checkpoint: saved `best.pt` from completed `zrv2yfgt`, not last.
- Policy and feature extractor: raw step1110, never EMA; no updates/sampling.
- The source is verified through the existing condition-width preflight,
  saved score replay, raw checkpoint provenance, and fixed K64 IDs.
- Filtered train: 416,701 events in
  `/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train`.
- Filtered external evaluation: 119,002 events in
  `/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val`.
- Complete filter manifests, actual parquet row counts and shape metadata are
  checked. No fallback to unfiltered data. All upstream normalization stays
  frozen; no input representation/feature additions.

Residual fitting uses ONLY the 62,213 old internal validation events, newly
hash-split 80/20 (about 49,770 fit and 12,443 validation). **None of the
354,488 first-stage gradient-training events is reused to fit the residual.**
The exact counts are logged; hashing is independent of labels, scores and Cij.

Important limitation: the first-stage best checkpoint was selected using the
old validation pool. This is not strict out-of-fold independence or a pristine
holdout. This fast experiment avoids in-sample gradient-training reuse and
avoids retraining the baseline, but retains base-selection dependence. A
positive result would warrant confirmation with fully independent/cross-fit
source fitting. It does not silently claim the full 416,701 events train the
new residual head.

After the residual checkpoint is frozen, evaluate Cij on all 119,002 external
conditions with their saved K64 candidates. Two NEW audit heads use a separate
hash 60/20/20 split within the external K1 population, approximately
71,401/23,800/23,800 conditions. They distinguish truth from first-stage-
weighted versus cumulative-weighted samples respectively. Both use identical
initialization/architecture/splits/budget; only negative weights differ.
Audit fits never change the ratio stack or select its checkpoint. Both audit
test sets are disjoint from their own fitting/selection and all ratio fitting.
The external pool was inspected in earlier experiments, so results remain
exploratory, not a pristine confirmatory test.

## Best validation is always restored

Every new head, including BOTH fresh audits:

- fresh initialization, AdamW 2e-4 -> 1e-5 with cosine over 500 epochs;
- weight decay .001; existing dropout; no gradient clipping added;
- minimum 1,000 training updates; then patience25, min_delta1e-4;
- **every strict validation weighted-BCE improvement saves best.pt**, including
  changes smaller than min_delta and checkpoints before the minimum updates;
- min_delta controls only patience. The final epoch, highest AUC, best Cij,
  and training BCE never choose the checkpoint;
- all endpoint inference explicitly loads the saved best.pt.

Log total fit steps AND selected checkpoint steps separately. An early
best-val checkpoint remains the selected checkpoint, but near-chance audit
scores with selected_steps<1000 are not treated as strong closure evidence.
No rerun with a different seed or repeated first-stage BCE control is added.

## Execution and W&B

All fitting, validation, frozen feature replay and scoring use 16 GPUs.
Batch size is up to 1024 paired conditions/GPU (1024 positive + 1024 negative
examples). The last global batch uses masked padding: every real event has
exactly one contribution. Feature extraction uses microbatches256. There is
no rank-zero full-pool validation bottleneck. Whole-event bootstrap is CPU
arithmetic and does not require GPU inference.

On the user's already-running Ray cluster, from `ml_pipeline`:

```bash
shifter python3 -u scripts/run_tau_residual_ratio.py \
  config/conditional_tau_residual_ratio_10pct.yaml
```

Optional read-only input preflight, without Ray/training/inference:

```bash
shifter python3 -u scripts/run_tau_residual_ratio.py \
  config/conditional_tau_residual_ratio_10pct.yaml prepare
```

`--ray-address "$RAY_ADDRESS"` is optional; the launcher reads it by default.
One new online W&B run is created, with separate epoch axes for `residual`,
`audit_baseline`, `audit_residual`. Display name:

**Can a residual classifier repair spin closure? | FiLM context256 | cumulative cap30**

No existing W&B runs are changed. Outputs have a unique directory under
`/pscratch/sd/y/yiren/Ztautau/conditional_tau_residual_ratio_1110`.
Large arrays and checkpoints remain on scratch; small reports/plot are uploaded.
No uploads, allocation requests or remote job submissions are performed by
the implementation assistant.

## Predeclared interpretation

Primary endpoint: residual-minus-baseline **off-diagonal Cij error on K64**,
paired whole-event bootstrap (2000 draws). Report all nine components, their
family9 simultaneous intervals, diagonal and total errors. Preserve the
adopted inverse-analyzing-power Cij convention; this experiment does not
re-certify that physics convention.

Report candidate/event ESS, max weight mass, grouped condition mass TV, and
uncapped versus capped product health. Endpoint guardrails: retain >=90% of
baseline candidate/event ESS, massTV increase<=.005, diagonal error
increase<=.02, total error increase<=.01. These are evidence flags, NOT a
Cij-based checkpoint selector or a hidden baseline fallback.

Fresh K1 audits: compare held-out weighted AUC gap and BCE using fixed-head
paired bootstrap500. A smaller AUC gap and BCE toward log(2) support removal
of detectable residual. A lower BCE for the residual TRAINING classifier is
not itself closure: it says that head found residual signal.

- Spin improvement + weaker fresh weighted discrimination: supports useful
  residual correction on available support; not proof of full closure.
- Better weighted BCE but no Cij improvement: discrepancy detection still does
  not yield a sufficiently useful ratio for the spin endpoint.
- Stronger uncapped tail concentration with cap suppressing the correction:
  ratio/support tradeoff remains; do not call iteration a general solution.
- Training improvement without validation improvement: do not deploy last;
  use the best-val artifact and report lack of generalization.

`stack.json` is sealed before external physics evaluation. It records both
checkpoints, log(Z1), cap, and selection metadata. `load_stack` in
`scripts/tau_residual_ratio.py` loads this exact composition on the existing
preprocessed inputs. No automatic DGPO deployment is performed.

Reference: [OmniFold](https://arxiv.org/abs/1911.09107), especially the
multiplicative weighted-classifier updates. This is an OmniFold-inspired
same-space residual ratio experiment, not the full detector/particle push-pull
unfolding algorithm.
