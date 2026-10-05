# Conditional tau-versus-spin ratio test on the 10% OmniFold data

The primary arm asks whether a classifier seeing the *tau pair itself* and the
visible condition can learn a useful conditional truth/generated ratio for the
raw step-1110 policy. The optional spin arm differs only in its candidate
representation. Neither arm changes DGPO or retrains the generator.
It uses the existing filtered 10% OmniFold train parquet (416,701 rows before
any mask filtering) for fitting and early stopping. The filtered diffusion
validation parquet is an independent evaluation pool. Source IDs must not
overlap; preparation fails if they do.

Both arms receive a packed visible EveNet condition and a 16-way decay
category indicator in one MLP branch. In the primary tau arm, the other branch
receives the directions of both reconstructed tau candidates (six numbers)
and their nine pair products. `tau_from_deltas` fixes the tau energy and
momentum magnitude, so these 15 numbers retain the candidate pair under the
generator's output convention without using `Cij`, analyzing powers or truth
spin labels. The optional spin arm instead receives 24 features:
the two three-component tau rest-frame visible directions, their nine outer
products, and the nine signed per-event terms `a_i*b_j/(kappa_a*kappa_b)`
whose weighted means (times 9) form the adopted Cij estimator. Positive and
negative examples share the same event condition and analyzing powers.
Truth and generated directions follow the same `tau_from_deltas` and `angles`
map in `scripts/diagnose_ztautau_cij.py`. No saved H4 backbone or reward
weights are loaded into the new classifier.

The spin inputs make the Cij ingredients explicit; BCE still optimizes density
classification, not Cij closure. Improved AUC alone cannot establish improved
Cij. The held-out tau-pair direction moments, per-channel spin moments and Cij
errors decide that. Even moment closure does not prove full conditional-density
closure; it is a necessary diagnostic, not a sufficient one.

Balanced, event-weighted BCE with one truth and one generated candidate per
condition estimates a ratio on the selected conditional candidate space. The raw
candidate log ratio is the **generated candidate's classifier logit**; the
truth logit is used only to report classification metrics. No tempering or clipping
is applied. The 10% OmniFold train pool is split by stable event identity into
85% fit and 15% early-stop events. Normalization uses fit events only. The
external validation pool is never used for fitting, normalization or early
stopping. The original saved H4 score is evaluated on this same external pool
as an exploratory historical reference; its prior fit overlap is not certified.

The classifier runs with one Ray Torch worker per GPU, distributed gradients,
AdamW and a cosine LR decay. The default budget is 250 epochs with early-stop
patience 30 after 1000 synchronized optimizer updates. The best validation
BCE checkpoint is scored on the independent pool. W&B receives training BCE,
validation BCE/AUC, test BCE/AUC, direct angular-moment closure, per-channel
closure and ESS. JSON reports and test scores remain in the output directory.

## NERSC commands

`config/conditional_tau_ratio_10pct.yaml` is the single experiment config. Its
train and validation paths are checked against the filtered 10% DGPO config
before any work starts. The previously generated one-candidate-per-condition
sample set under `h4_step1110_tau_train_20260930` is reused. After syncing
the updated repository into the existing `ml_pipeline`, first verify the
saved samples and both parquet pools in Shifter on an allocated compute node:

```bash
shifter python3 -u scripts/run_conditional_tau_ratio.py \
  config/conditional_tau_ratio_10pct.yaml prepare
```

This CPU preflight does not connect to Ray or start W&B. It compares full
sample identities, packed visible conditions and truth targets against the
parquet, and confirms matching raw checkpoint and DDIM setup. Only if it
prints `"prepared": true`, use the usual 16-GPU Ray allocation and run:

```bash
shifter python3 -u scripts/run_conditional_tau_ratio.py \
  config/conditional_tau_ratio_10pct.yaml train
```

Training gets its own W&B run. If samples must be regenerated later, set a
fresh `train_sample_root` in the YAML and run its `sample` phase on the
16-GPU allocation first. A completed sample set under the configured root
prevents accidental duplicate generation.

If a matched spin-representation control is useful, use the exact same saved
samples and settings but a fresh output directory:

```bash
shifter python3 -u scripts/train_conditional_spin_ratio.py \
  --train-source /pscratch/sd/y/yiren/Ztautau/h4_step1110_tau_train_20260930 \
  --test-source /pscratch/sd/y/yiren/Ztautau/h4_film_step1110_cij/rescore-ca89b70725 \
  --workers 16 --batch-size 256
```

The training launcher discovers a sample under the train source root only if
there is exactly one completed sample directory. If there are multiple, pass
the specific `sample-*` directory. It checks that train and test samples use
the same raw checkpoint, DDIM steps, candidate count and weight mode before
training. The YAML explicitly supplies the filtered 10% DGPO validation
directory. If the test manifest records `event_source`, it must match. This
historical manifest does not, so preparation rebuilds the packed visible
conditions and truth tau targets from parquet, requiring event-by-event
agreement with the saved candidate bundle before training. The assistant
does not submit the allocation or training job.

The primary decision is whether held-out tau-pair moment error improves under
the tau ratio with acceptable ESS, overall and within decay-channel and
visible-pT strata, and whether independently calculated channel Cij errors
also improve. A failed tau-ratio reweighting test would
localize the problem before DGPO; a successful fixed-sample test would only
justify testing reward transfer in DGPO, not claim generator closure.

## Matched normalization ablation after qelc17rx

Use `config/conditional_tau_ratio_masked_norm_10pct.yaml`. The question is
whether per-slot statistics contaminated by padding caused the extreme raw
ratios. The historical run qelc17rx is the control; it is not retrained.
The new config pins its completed training sample directory, checks the
control manifest's optimization settings and exact event splits, and trains
the same fresh MLP with the same seed and 16 workers.

Only classifier condition preprocessing changes: x statistics pool the same
feature over valid fitting particles across slots, binary features and masks
retain their values, and padded slots are zero after normalization. Global
condition statistics ignore masked events. The existing [-20,20] input cap
is retained. No new log/CDF/Fourier transformation is added. This shares
EveNet's mask-aware feature-wise design, but does not load or change the
generator's pinned normalization. Other continuous condition fields keep
fit-only per-coordinate statistics. Mean, scale, packing spec and mode are
saved with the best classifier checkpoint for reproducible inference.

```bash
shifter python3 -u scripts/run_conditional_tau_ratio.py \
  config/conditional_tau_ratio_masked_norm_10pct.yaml train
```

The train phase includes all preflight checks; `prepare` remains available
for a CPU-only data check. A new W&B run records per-epoch validation ratio
ESS, maximum weight mass, maximum logit and log mean ratio alongside BCE/AUC.
These are diagnostics and do not select checkpoints: minimum validation BCE
still determines best.pt. The independent test is evaluated only afterward.

Endpoint reports include normalized-input clipping/padding counts, ratio
normalization and weight fractions by decay category, particle count and
their intersections, and tau/angular/Cij moments. `candidate_ratio/*` denotes
the new classifier; `previous_tau/*` replays qelc17rx scores on exactly the
same test events; `saved_h4/*` is the older H4 reference. The misleading legacy
`spin_only/*` name is no longer used for tau runs. Saved test scores are also
uploaded so tail analysis can be done without another remote training run.

The primary physics endpoint is the held-out matched Cij error relative to
unweighted samples, overall and by channel. Improvement over a catastrophically
bad control alone is insufficient. Check tau moments, condition-mass drift and
weight support as additional diagnostics. If tails stabilize but Cij remains
worse, normalization was not a sufficient solution. Bootstrap intervals are
conditional on the fixed fit/samples and are unreliable at near-one ESS.
This experiment does not train an additional independent judge and does not
claim full distribution closure from a finite set of moments.
