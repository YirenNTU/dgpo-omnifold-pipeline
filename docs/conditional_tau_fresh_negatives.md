# Fresh training negatives, fixed FiLM conditional tau classifier

## Evidence and one question

Existing FiLM `pzq0nl1i` classified well but raw reweighting did not close Cij.
On the fixed fresh-K64 panel `fzekzrmr`, Cij norm errors were about 0.56544
unweighted, 0.81752 raw, and 0.36580 with cap30. Component diagnostic
`mid2yoj4` showed cancellation between helpful and harmful contributions;
it did not prove that a tiny tail alone causes the error. Ordinary validation
overfitting, insufficient conditioning, and fixed-negative reuse are not
established causal explanations.

Question: does replacing the same cached training negative every epoch with
a fresh draw improve the learned raw conditional ratio and held-out Cij?
This tests negative-sample reuse, not a new architecture or physics prior.

## Matched design

- Reuse the original filtered train416701 and validation119002 populations,
  with identical fit354488 / internal-val62213 / external-test119002 splits.
- Frozen raw step1110 generator and frozen raw1110 candidate-feature backbone.
  No EMA. Same DDIM20 and checkpoint normalization.
- Same three-block FiLM head, fresh initialization seed42, unchanged tau15,
  relative6 and cached decoder representation. No Cij/analyzing power inputs.
- Same paired BCE, one truth and one negative per condition each epoch,
  same event weight on both classes. Thus no class-prior correction: ratio
  remains exp(logit), with global normalization only for expectations.
- AdamW2e-4, cosine to1e-5 over250 epochs; weight decay0.001, dropout0.05;
  patience25, min_delta1e-4, minimum1000 optimizer steps; checkpoint selected
  solely by lowest internal-val BCE. These are inherited and checked, not tuned.
- 16 GPUs,1024 conditions/GPU. Each negative is generated just before its batch;
  a frozen backbone extracts its features in microbatches256. Independent
  generation RNG leaves classifier initialization/dropout unchanged. Same
  DistributedSampler padding as control (8 duplicate fit slots/epoch) is logged.
- Fixed validation and test candidates, shared frozen preprocessing. Refresh
  occurs only for fit rows. Generator initialization and historical feature
  replay are checked before updates. No score-dependent candidate selection.

Sampling arithmetic follows the historical training-sample worker's default
`highest` float32 matmul precision, not the unused DGPO runtime precision field.
The K64 panel is already saved; its sampling realization is unchanged.

The compute budgets are not wall-clock matched: DDIM20 plus feature extraction
is repeated each epoch. Truth samples remain fixed, so this is not a complete
solution to conditional-ratio estimation or proof that data coverage is fixed.

## Endpoints and W&B

Training/validation BCE,AUC, LR, ESS, max weight, conditional diagnostics,
early-stop state, fresh-negative count, generation/extraction time, feature
replay error, sampler padding. Existing rank-zero training validation is
preserved; final K1 and K64 inference use all16 GPUs without padding.

After training: score the same stored119002×64 candidates from `fzekzrmr`
with the selected fresh-negative head. Do NOT resample the panel. Check old
classifier replay, then compare unweighted, old raw, old cap30, fresh raw,
fresh cap30. Cap30 is fixed evaluation sensitivity, never a training loss or
checkpoint selector. Store all nine Cij entries, errors, paired event-bootstrap
intervals, candidate/event ESS, log mean ratio and maximum candidate weight.

Primary contrast: fresh raw minus old raw Cij norm error. Also compare against
unweighted and the stronger old cap30 benchmark; do not call a reduction
relative to a poor raw baseline full closure. Component intervals are
pointwise and exploratory; fixed-head bootstrap excludes fit uncertainty and
unseen tails. This panel's conditions have been inspected before.

## User launch (existing Ray allocation)

```bash
shifter python3 -u scripts/run_tau_fresh_negatives.py \
  config/conditional_tau_fresh_negatives_10pct.yaml
```

Optional input/protocol preflight (no GPU training): append `prepare`.
Uses RAY_ADDRESS automatically. A new W&B run and fresh output folder are
created; no prior experiment is overwritten. Original fixed-negative FiLM
and cached BCE are reused, not trained again. This code does not submit jobs.

Validation: CPU protocol tests and mocked real training-worker integration;
production CUDA/Ray/DDIM end-to-end execution requires NERSC and has not been
run locally. In particular,16-GPU memory/performance is not certified by CPU tests.
