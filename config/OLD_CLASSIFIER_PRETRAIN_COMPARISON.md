# Old classifier on 10% pretrain versus pair-token on step1110

User requested comparison, 2026-09-21. Truth remains the clean validation pool
`diffusion_val_20pct_seed42_stic_filtered_test1/val` for both arms. The 10%
label denotes the pretrained diffusion lineage, not a new truth subsample.

## Question and scope
Which differences does each classifier learn, and how concentrated/useful are
its held-out ratios? Existing h4pair01 is complete on step1110. New oldpre01
fits the historical no-pair-feature classifier on raw 10% pretrained diffusion.
This comparison changes both generator and classifier recipe. It cannot isolate
an architecture effect or claim that lower AUC means a better classifier.
No policy training, reward installation, truth retuning or test-based selection.

## Frozen contract and provenance
User: new generator and classifier backbone source is
`/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt`.
Record the loaded checkpoint global_step at preflight; last.ckpt is mutable.
Load policy weights-only with fresh experiment state, no EMA substitution.
User: 16 GPUs, existing allocation. User: original clean validation truth pool.
Historical recipe owner c4a91e07: no Fourier, pair token, theta-pair features or
rest-frame branch. Two-token decoder128, one layer, four heads, dropout .25.
PET adapters, GroupedSequentialEmbedding and InvisibleInputProjector train;
last PET block and other backbone stay frozen. Head/adapter/decoder LR2e-4,
backbone LR1e-5, weight decay .0005, clipping5, constant AdamW.
Measurement owner h4pair01: identity-disjoint fit/early-stop/test, K1, same
pool selection seeds and16 workers,1000 minimum/3000 maximum updates,
ten-epoch BCE patience, min_delta .001, restored-best BCE model. This stopping
budget is intentionally more controlled than the historical min_steps1.
Pair context is packed only for diagnostic export, not passed as model features.

## Measurements and decision
Save classifier and held-out events. Read test BCE/AUC, fit steps/best step,
full learning/gradient/clipping curves, raw and fixed-.75 ESS/N, top1% and max
mass, mean ratio and uncertainty, top20 event coordinates, input validity,
removal sensitivity, fine angular topology/Wasserstein diagnostics.
Report confidence intervals where the existing exporter supplies them.
Compare rank concentration and affected regions, not only headline AUC.
No binary architecture winner: different policy distributions preclude it.
A usable ratio should improve the measured held-out projections without
concentration-driven deterioration; this remains a projection diagnostic,
not full closure. Do not optimize temperature/clipping on this test.

## Run
W&B `oldpre01`; name `What did the old classifier learn? | 10% pretrain | held-out ratio audit`;
group `Classifier signal comparison`.

```bash
shifter python3 scripts/train_old_classifier_pretrain.py --check-only
shifter python3 scripts/train_old_classifier_pretrain.py
```
No automatic extra seed or extra architecture arm. Refuse existing outputs.

## Result table
| Run | Generator | Test BCE/AUC | Fit/best updates | ESS | top1%/max mass | acoplanarity/acollinearity W1 change |
|---|---|---|---|---|---|---|
| h4pair01 | step1110 | .379366 / .899804 | 1048 / 1008 | 1672.7 (6.99%) | 27.86% / .2040% | .009677→.002723 / .008422→.004490 |
| oldpre01 | 10% pretrained raw, global_step13975 | .629472 / .701783 | 1036 / 504 | 4771.7 (19.94%) | 16.42% / .2363% | .010517→.011310 / .008488→.010414 |

## Musk gate
Link: discrimination and ratio usability. Requirements: user sources/truth/GPU
budget, c4a91e07 historical model recipe, h4pair01 diagnostic measurement budget.
Delete reward fits, policy updates, Fourier branch probes and parameter sweeps.
Fastest falsifier: one cold legacy fit plus saved-artifact analysis. No existing
saved legacy best-BCE artifact answers the requested new pretrain fit.
First valid final point at >=1000 updates; elapsed runtime is measured, not
assumed. Kill on nonfinite training/invalid inputs, cap3000. Cost16*elapsed h
per changed hypothesis; do not call a confounded architecture comparison a win.
Session scope overrides older registry priorities per explicit user request.

Completed analysis: `artifacts/classifier_signal_comparison/REPORT.md`. Both W&B runs finished. Old ratio export validated against saved logits despite Ray shutdown SIGSEGV after completion. Common K8 comparison not executed: allocation58683864 completed before scoring; no ordering/gradient claim.
