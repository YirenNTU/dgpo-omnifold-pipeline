# Does head conditioning improve conditional tau ratios?

## Question and frozen controls

MLC `s01j1pez` failed with test ESS 2.79 and matched Cij error 28.159.
nnUKL `sfisiaz2` reduced this to ESS 886.91 and error 2.023, but unweighted
error is 0.521 and saved BCE `443eg16h` error is 1.111. These show that tail
control alone was not sufficient; they do NOT isolate conditioning as the cause.

This experiment changes ONLY the trainable classifier head's fusion operation.
It never updates diffusion/DGPO, reloads EMA, regenerates samples, extracts new
features, introduces physics targets into the loss, caps ratios or changes inputs.
It reuses the exact `443eg16h/prepared.npz` population: 354488 fit, 62213 internal
validation, 119002 external test events, K=1, filtered 10% dataset. Checkpoint
source is raw step1110; its candidate-dependent pre-velocity representations
already contain visible context. This is an ablation of ADDITIONAL head
conditioning, not a comparison against a completely unconditioned classifier.

## Two new fits, with a reused historical anchor

Both arms retain the existing condition encoder (input -> 128 -> 64, SiLU)
and candidate encoder (cached features + relative angles + tau moments -> 64
-> 64, SiLU). Both have THREE residual fusion blocks, LayerNorm on the hidden
state, SiLU, dropout .05 and a final scalar logit. Residual scale is 1/sqrt(3).
At each block let z=LayerNorm(h), c be the encoded visible condition:

- **concat:** h += scale * Wout(dropout(SiLU(Wh*z + Wc*c + b))).
  This is exactly an affine layer on concatenated [z,c]. Context is supplied
  at every block, so depth and repeated injection are not exclusive to FiLM.
- **FiLM:** (gamma,beta)=Wc*c; h += scale *
  Wout(dropout(SiLU(Wh*((1+gamma)*z+beta)+b))).

Wh and Wc both map 64 -> 128 in both arms. Parameter tensors, counts, encoder
initialization and starting output functions are IDENTICAL across the two arms.
Wc starts at zero in both; the remaining weights use normal initialization.
There is no zero output gate, so Wc can learn on update one. Context-encoder
gradients begin once Wc moves away from zero. Runtime preflight records and
checks the exact trainable parameter counts using actual cached input widths.

The original shallow BCE head is reused as a historical anchor, NOT retrained.
The deeper concat is a necessary NEW capacity/depth control; comparing FiLM
only with old BCE would conflate fusion, depth, normalization and parameter count.

## Matched training and selection

- 16 Ray GPU workers, 1024 paired conditions per GPU (1024 positive and 1024
  generated examples). Global nominal batch: 16384 conditions.
- Fresh heads, same seed42, BCE only, balanced paired weights. Frozen raw trunk.
- Inherit verified baseline AdamW LR2e-4, weight decay .001, 250-epoch cosine
  decay to1e-5. Max250 epochs; patience25, min_delta1e-4, min1000 updates.
- Absolute lowest internal validation BCE selects best.pt; min_delta governs
  patience only. This deliberately holds the selector fixed, not a claim that
  BCE-only selection ensures good ratios. No Cij/test-driven checkpoint choice.
- The legacy zero-coefficient training MMD diagnostic is skipped in BOTH arms;
  validation MMD remains. No MMD training loss is introduced.
- Inference uses all16 GPUs, exactly-once event-ID checked shards, no padding.

## W&B measurements and decision

Every epoch: train/validation BCE, AUC, LR, ESS, max mass, log mean ratio,
max log ratio, early-stop state, learned context-projection weight norm;
`val_ratio_risk` is the uncorrected .5*(E_generated exp(logit)-E_truth logit)
evaluated as a diagnostic even though training uses BCE.

Every5 epochs, plus first/final: `val_condition/*` on the full internal
validation pool. Visible-pT quartile edges come ONLY from the fit split, and
are shared across arms. Evaluate categories, pT bins and their intersections:

- raw log mean ratio in each group (ideal population value0 under overlap),
- original vs globally reweighted group probability mass,
- within-group ESS and tau direction/product moment error.

These are diagnostics, not per-event normalization. K=1 cannot estimate a
pointwise conditional expectation. Group moment means self-normalize within
groups for reporting only; original raw weights remain unchanged in Cij.
No Cij labels, analyzing powers or test data enter training or fit-only strata.
These group diagnostics are point estimates and can fluctuate with heavy tails;
they cannot alone separate systematic ratio bias from sampling variance.
The separately retained legacy tau-moment report uses its original external
analysis-only pT quartiles for historical comparison. The new conditional
report and live conditional diagnostics always use the shared fit-only cuts.

At the selected endpoint, save test_scores.npz, conditional_ratio_report.json,
tau moment reports, all nine Cij components and paired event-bootstrap (500)
differences in Cij Frobenius error. FiLM automatically compares with the
completed same-protocol concat pointer when present and records that run ID.
Historical BCE and unweighted comparisons are retained in both arms.

**Primary physics gate:** negative Cij error difference against BOTH unweighted
and matched concat, with paired interval upper bound <0. Do not call lower
error than a bad concat alone successful alignment. Supporting diagnostics:
conditional mass/ratio errors and tau moments should improve without renewed
catastrophic tails. AUC improvement alone fails the question. If no matched
concat completed, the FiLM-vs-concat attribution is unavailable, explicitly logged.

The external test has been inspected in previous rounds: this is exploratory.
Bootstrap is fixed-model/fixed-candidate, not refit uncertainty, and is fragile
at very low ESS. Adopted Cij convention stays unchanged; physical polarimetry
and acceptance are not newly certified. No fresh weighted audit is automatically
trained here, and these finite moments do not prove full conditional closure.

## User launch commands

On the existing NERSC checkout/allocation, after syncing the changed files:

```bash
shifter python3 -u scripts/run_tau_conditioning.py \
  config/conditional_tau_conditioning_10pct.yaml all
```

Runs concat then FiLM sequentially, each with16 GPUs and its own new online
W&B run; old outputs are untouched. `prepare` checks sources/parameter counts
without training or Ray. `concat` / `film` run one arm. No jobs are submitted
by this implementation. Repository-root uploads must use
`--exclude-from=NERSC/upload-excludes.txt` and must exclude toy outputs.

## Executed concat control — 2026-09-30

[69rj3rmc](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/69rj3rmc),
`Does conditioning fix tau ratios? | matched concat | 3 blocks | raw 1110`,
finished all250 epochs /5500 updates without early stopping. The selected
checkpoint is epoch250. Runtime verifies187649 trainable parameters in each
new head. This is the concat arm, not FiLM. The companion `pzq0nl1i` was
running at inspection; its outcome is not yet established.

Full trajectory and uploaded `cij_comparison.json` plus
`conditional_ratio_report.json` were inspected. Validation BCE declines
0.40648/0.37912/0.31727/0.26997 at epochs25/50/100/250, while validation
ESS fractions decline12.57%/9.85%/1.81%/0.90%. The final held-out test BCE
0.271778 and AUC0.954830 closely match internal validation0.269975/0.955378.
Old BCE test scores were0.390485/0.903227. Classification generalization is
materially better; this is not evidence of improved generator samples.

Same119002-event test endpoints versus saved shallow BCE:

| Metric | Old BCE | Deep concat |
| --- | ---: | ---: |
| ESS |903.30|1654.66|
| ESS fraction |0.759%|1.390%|
| Max mass |3.049%|0.716%|
| Top1% mass |30.747%|52.280%|
| Generated raw mean ratio |1.0894|1.04338|
| Category mass TV |0.04741|0.03451|
| Group log-mean-ratio RMS |0.19519|0.20516|
| Group tau moment error |0.13543|0.22271|
| Matched Cij error |1.11056|1.78144|

Unweighted group tau error is0.003332; unweighted Cij error is0.521308.
Concat-minus-unweighted Cij error is+1.260134, paired95% interval
[0.765190,2.674628]. Concat-minus-old-BCE is+0.670881 with interval
[-0.498580,2.060888], so the apparent worsening relative to old BCE is not
resolved by this bootstrap. Fixed-model/bootstrap and previously inspected
test limitations still apply.

Conclusion: extra depth/residual repeated additive conditioning substantially
improves classification but is not sufficient for matched physics alignment.
Lower max weight and better total normalization coexist with worse group
moments and substantial top1% mass. Do not call the whole tail uniformly
better or explain failure solely by one largest event. This arm bundles depth,
normalization and fusion changes versus historical BCE; only the companion
capacity-matched FiLM can isolate the fusion operation. No fresh weighted
audit was performed.

## Executed FiLM arm — 2026-09-30

[pzq0nl1i](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/pzq0nl1i),
`Does conditioning fix tau ratios? | matched FiLM | 3 blocks | raw 1110`,
finished 250 epochs / 5500 updates; minimum-validation-BCE checkpoint is
epoch250, no early stop. Full 250-row history and uploaded Cij/conditional
reports were inspected. Runtime confirms matched concat `69rj3rmc`, identical
187649-parameter heads, same cached raw1110 samples, fit354488/val62213/test119002,
16 GPUs, 1024 paired conditions/GPU, and no backbone or policy updates.

| Held-out metric | Matched concat | FiLM |
| --- | ---: | ---: |
| Test BCE |0.271778|0.267325|
| Test AUC |0.954830|0.956313|
| ESS fraction |1.39045%|1.01747%|
| Top1% weight mass |52.2797%|54.5206%|
| Maximum weight mass |0.71578%|0.74644%|
| Category mass TV |0.03451|0.05238|
| Group log-mean-ratio RMS |0.20516|0.23599|
| Group tau moment error |0.22271|0.24086|
| Cij Frobenius error |1.78144|1.42241|

FiLM-minus-concat Cij error is -0.35903, paired95% CI[-0.90337,0.75682]:
improvement is unresolved. FiLM-minus-unweighted is +0.90110,
CI[0.67909,2.63298]; unweighted error0.52131. The primary physics gate fails.
Internal validation BCE falls0.40687/0.37396/0.31316/0.26519 at
epochs25/50/100/250, while ESS falls11.56%/7.99%/1.52%/0.546%.
Training BCE0.26939 and external testBCE0.26733 track validation0.26519;
there is no conventional widening BCE generalization gap at the endpoint.

Conclusion: this capacity-matched nonlinear FiLM fusion slightly improves
classification point estimates, but does not deliver usable spin-correlation
reweighting. It rules out this head-fusion change as a sufficient solution,
not conditioning limitations in general. Falling ESS alone is not proof of
incorrect ratios; finite-sample support/tail variance and ratio bias remain
unseparated. Conditional diagnostic point estimates worsen despite lower BCE.
No new generator was trained and no fresh weighted audit was performed.
Paired intervals condition on the trained models and generated candidates;
the repeatedly inspected test is exploratory, not a new confirmatory holdout.
