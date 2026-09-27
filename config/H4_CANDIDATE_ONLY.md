# H4 direct candidate readout without engineered pair features

## Question
Can the original two-candidate path learn a generalizable truth-versus-generated
signal without explicit pair-angle Fourier features? Requested after h4rel01's
early train/validation divergence (at 300 updates: train BCE .6413, validation
BCE .7528, validation AUC .5073). This early observation does not establish the
final relation-model outcome or absence of a learnable signal.

## Architecture
Use the existing EveNet/PET candidate-conditioned representation. The decoder
has two candidate tokens, one block, hidden width 128 and four attention heads.
Retain self-attention, visible-memory cross-attention, event-conditioned AdaLN,
FFN and output LayerNorm. Flatten both token outputs (256 dimensions) into the
existing zero-initialized linear logit head. There are no relation latents,
mean-pooling bottleneck, engineered pair features, Fourier branch, rest-frame
branch, auxiliary physics losses, projections, smoothing or shuffled negatives.

This reuses the tested legacy decoder; no new model implementation is needed.
Compared with h4pair01, remove the engineered third token and its readout slot.
Compared with h4rel01, depth/readout/initialization also change; do not attribute
any improvement solely to pooling or Fourier. It does not guarantee high-order
structure learning. Candidate features may still encode fine angular effects.

## Fixed training contract
- Classifier backbone: /pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt
- Generator only: /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/checkpoints/dgpo-epoch=110-next_ep=111-step=1110.ckpt
- Truth: /pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val
- Fresh weights-only generator state, K=1 generated pool, unchanged sampler,
  seed and identity-disjoint fit/early-stop/final-test splitting protocol.
- 16 GPU workers, one GPU each. Existing Ray cluster must have >=16 GPUs;
  the launcher never creates a cluster or requests an allocation.
- Train adapters, decoder/head and last PET block, matching h4pair01.
- Preserve all training/normalization settings: head LR 2e-4, decoder/adapters
  5e-5, last PET block 1e-5, weight decay .001, dropout .25, clipping 5.
- 1000 minimum / 3000 maximum fit updates; constant LR; ten effective epochs
  validation patience, min_delta .001. Best checkpoint selected by validation
  BCE; independent final test is never used for selection.
- Export best classifier and test scores for later BCE/AUC, Brier/ECE, ESS and
  weight concentration analysis. No policy updates and no reward installation.

## Decision and interpretation
Primary: independent test BCE at the minimum-validation-BCE checkpoint.
Report validation AUC and train/validation gap at shared update counts (especially
100/200/300/1000), best/actual steps, clipping and convergence state. AUC near
chance or lower AUC alone is not success. A cap-limited fit is not saturation.
Use exact saved histories rather than W&B summaries for aligned comparisons.
Fixed protocol does not imply bit-identical regenerated candidates.

## Interactive launch (user submits all jobs)
After updating /global/homes/y/yiren/ml_pipeline and preparing the interactive
16-GPU Ray cluster in the Python environment with the training dependencies:

```bash
cd /global/homes/y/yiren/ml_pipeline
export PYTHONPATH="$PWD/evenet_dgpo:$PWD/scripts:$PWD:${PYTHONPATH:-}"
python3 scripts/train_h4_candidate_only.py --check-only
python3 scripts/train_h4_candidate_only.py
```

Local config-only check: `python3 scripts/train_h4_candidate_only.py --validate-only`.
An existing output directory is rejected; use `--run-suffix retry1` for a fresh
explicit retry. W&B ID h4cand01; outputs h4_candidate_only under Ztautau pscratch.
No job has been submitted by the assistant for this experiment.
